# -*- coding: utf-8 -*-
import os, json, warnings
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_erosion
from skimage.morphology import skeletonize_3d, remove_small_objects, ball
import matplotlib.pyplot as plt

# =========================
# 标签定义（新的左右分区）
# =========================
# 软骨（cartilage）
LF, MF, LT, MT, LP, MP = 201, 202, 203, 204, 205, 206
# 软骨下骨带（subchondral bands, 10mm 或你的带宽）
LF_SUB, MF_SUB, LT_SUB, MT_SUB = 211, 212, 213, 214

# 可选：如果你的“总 ROI 图”里还会出现旧编号，可在此做兼容映射
LEGACY_IGNORE = {0}  # 背景等需要忽略的标签

# 供遍历用的清单（名称用于文件与表格）
CART_ROIS = [
    (LF,     "lf_cartilage"),
    (MF,     "mf_cartilage"),
    (LT,     "lt_cartilage"),
    (MT,     "mt_cartilage"),
    (LP,     "lp_cartilage"),
    (MP,     "mp_cartilage"),
]
SUB_ROIS = [
    (LF_SUB, "lf_subchondral_10mm"),
    (MF_SUB, "mf_subchondral_10mm"),
    (LT_SUB, "lt_subchondral_10mm"),
    (MT_SUB, "mt_subchondral_10mm"),
]
ALL_ROIS = CART_ROIS + SUB_ROIS

# =========================
# I/O 辅助
# =========================
def load_nifti(path):
    nii = nib.load(path)
    arr = np.asanyarray(nii.dataobj)
    return arr, nii.affine, nii.header, np.array(nii.header.get_zooms()[:3], float)

def save_nifti(arr, affine, header_like, path, dtype=np.uint8):
    nib.save(nib.Nifti1Image(arr.astype(dtype), affine, header_like), path)

def write_single_mask_like(total_roi_path, target_label, out_mask_path):
    arr, aff, hdr, _ = load_nifti(total_roi_path)
    m = (arr == target_label).astype(np.uint8)
    save_nifti(m, aff, hdr, out_mask_path)
    return out_mask_path

# =========================
# 形态学度量
# =========================
def clean_mask(mask, min_vox=100, radius=1):
    m = remove_small_objects(mask.astype(bool), min_size=min_vox)
    if radius > 0:
        m = binary_erosion(m, structure=ball(radius), iterations=0).astype(bool) | m
    return m

def morphology_metrics(mask, voxel_mm):
    mask = mask.astype(bool)
    voxvol = float(np.prod(voxel_mm))
    volume_mm3 = mask.sum() * voxvol
    if mask.sum() == 0:
        return dict(volume_mm3=0.0, mean_thickness_mm=0.0, p95_thickness_mm=0.0)

    dist = distance_transform_edt(mask, sampling=voxel_mm)
    skel = (skeletonize_3d(mask) > 0)
    dvals = dist[skel] if np.any(skel) else dist[mask]
    thickness = 2.0 * dvals
    return dict(
        volume_mm3=float(volume_mm3),
        mean_thickness_mm=float(np.mean(thickness)) if thickness.size else 0.0,
        p95_thickness_mm=float(np.percentile(thickness, 95)) if thickness.size else 0.0,
    )

# =========================
# PyRadiomics
# =========================
def default_pyradiomics_params():
    return {
        "imageType": {
            "Original": {},
            "LoG": {"sigma": [1.0, 1.5, 2.0, 2.5]}
        },
        "setting": {
            "normalize": False,
            "binWidth": 25,                 # 可被 run_case(bin_width=) 覆盖
            "resampledPixelSpacing": None,
            "interpolator": "sitkBSpline"
        }
    }

def extract_radiomics(image_path, mask_path, label=1, params=None):
    from radiomics import featureextractor

    if params is None:
        params = default_pyradiomics_params()

    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    try:
        extractor.addProvenance(False)
    except Exception:
        pass

    # 兼容性：尽量开启所有可用特征
    ok = False
    try:
        extractor.enableFeaturesByName(
            firstorder=[], glcm=[], glrlm=[], glszm=[], gldm=[], ngtdm=[], shape=[]
        ); ok = True
    except Exception:
        pass
    if not ok:
        try: extractor.enableAllFeatures(); ok = True
        except Exception: pass
    if not ok:
        for cls in ["firstorder","glcm","glrlm","glszm","gldm","ngtdm","shape"]:
            try: extractor.enableFeatureClassByName(cls)
            except Exception: pass

    try:
        extractor.settings["enableCExtensions"] = True
    except Exception:
        pass

    result = extractor.execute(image_path, mask_path, label=label)

    # 仅保留可转成 float 的项，并去掉 diagnostics_*
    feats = {}
    for k, v in result.items():
        if str(k).startswith("diagnostics_"):
            continue
        try:
            feats[k] = float(v)
        except Exception:
            continue
    return feats

# =========================
# 可视化（热力图）
# =========================
def save_t2_heatmap_png(image_path, total_roi_path, out_png, vmin=None, vmax=None, axis='sag', idx=None):
    import logging
    logging.getLogger('radiomics').setLevel(logging.WARNING)

    img, _, _, _ = load_nifti(image_path)
    roi, _, _, _ = load_nifti(total_roi_path)
    roi_bool = roi > 0

    if axis == 'sag':
        dim = 0
    elif axis == 'cor':
        dim = 1
    else:
        dim = 2

    if idx is None:
        if dim == 0:
            sums = roi_bool.reshape(roi_bool.shape[0], -1).sum(axis=1)
        elif dim == 1:
            sums = roi_bool.transpose(1,0,2).reshape(roi_bool.shape[1], -1).sum(axis=1)
        else:
            sums = roi_bool.transpose(2,0,1).reshape(roi_bool.shape[2], -1).sum(axis=1)
        idx = int(np.argmax(sums))

    if dim == 0:
        sl_img = img[idx, :, :]
        sl_msk = roi_bool[idx, :, :]
    elif dim == 1:
        sl_img = img[:, idx, :]
        sl_msk = roi_bool[:, idx, :]
    else:
        sl_img = img[:, :, idx]
        sl_msk = roi_bool[:, :, idx]

    sl_img = sl_img.T
    sl_msk = sl_msk.T

    if vmin is None or vmax is None:
        vals = sl_img[sl_msk]
        if vals.size > 0:
            vmin_auto = np.percentile(vals, 5)
            vmax_auto = np.percentile(vals, 95)
            vmin = vmin if vmin is not None else float(vmin_auto)
            vmax = vmax if vmax is not None else float(vmax_auto)

    plt.figure(figsize=(6,6))
    plt.imshow(sl_img, origin='lower', cmap='gray')
    overlay = np.ma.array(sl_img, mask=~sl_msk)
    plt.imshow(overlay, origin='lower', alpha=0.6, vmin=vmin, vmax=vmax)
    plt.axis('off'); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

# =========================
# 主流程（单病例）
# =========================
def run_case(image_path, total_roi_path, out_dir, id_tag="case", bin_width=25):
    """
    参数：
        image_path: 原始图像（T2, .nii/.nii.gz）
        total_roi_path: “总ROI图”（已按 201–206/211–214 编码）
        out_dir: 输出目录
        id_tag: 病例ID前缀（用于文件命名）
        bin_width: PyRadiomics 的 binWidth
    输出：
        <id>_morphology.json
        <id>_radiomics.json
        <id>_features.xlsx / .csv  （一例一表：每 ROI 一行）
    """
    os.makedirs(out_dir, exist_ok=True)
    arr_img, aff, hdr, vsz = load_nifti(image_path)
    arr_roi, _, _, _ = load_nifti(total_roi_path)

    # ---- 形态学 ----
    metrics = {}
    for lbl, name in ALL_ROIS:
        m = (arr_roi == lbl)
        metrics[name] = morphology_metrics(m, vsz)

    # ---- Radiomics ----
    params = default_pyradiomics_params()
    params["setting"]["binWidth"] = bin_width
    feats = {}
    for lbl, name in ALL_ROIS:
        mask_path = os.path.join(out_dir, f"{id_tag}_{name}_mask.nii.gz")
        write_single_mask_like(total_roi_path, lbl, mask_path)
        feats[name] = extract_radiomics(image_path, mask_path, label=1, params=params)

    # ---- JSON 保存 ----
    with open(os.path.join(out_dir, f"{id_tag}_morphology.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, f"{id_tag}_radiomics.json"), "w") as f:
        json.dump(feats, f, indent=2)

    # ---- 合并为“长表”并导出 Excel/CSV ----
    rows = []
    for name in [n for _, n in ALL_ROIS]:
        row = {"id": id_tag, "roi": name}
        # 形态学
        for k, v in metrics.get(name, {}).items():
            row[f"morph_{k}"] = v
        # radiomics 展开
        for k, v in feats.get(name, {}).items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    xlsx_path = os.path.join(out_dir, f"{id_tag}_features.xlsx")
    csv_path  = os.path.join(out_dir, f"{id_tag}_features.csv")
    # 有些 radiomics 键名很长，Excel 列多属正常
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"[WARN] 写 Excel 失败：{e}")
    df.to_csv(csv_path, index=False)

    return metrics, feats

# =========================
# （可选）旧版函数占位
# =========================
def make_subchondral_band(*args, **kwargs):
    """保持向后兼容：若你仍需从“骨+软骨”生成骨带，可在此重新实现。
       但在当前左右编码工作流中，通常不再需要。"""
    raise NotImplementedError("当前工作流使用已分左右的 SUB 标签（211–214），不再在此构建骨带。")


