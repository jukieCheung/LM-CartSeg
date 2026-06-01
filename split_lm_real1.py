#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ===== 输入多标签 =====
SUB_FEMUR    = 10
SUB_TIBIA    = 11
CART_FEMUR   = 20
CART_TIBIA   = 21
CART_PATELLA = 22

# ===== 输出标签（不与原始冲突）=====
LF, MF, LT, MT, LP, MP = 201, 202, 203, 204, 205, 206
LF_SUB, MF_SUB, LT_SUB, MT_SUB = 211, 212, 213, 214


def _vox2world(idx, aff):
    """ idx: (N,3) voxel -> world (N,3) """
    homo = np.c_[idx, np.ones(len(idx))]
    xyz  = homo @ aff.T
    return xyz[:, :3]


def _auto_laterality_from_name(path, default='L'):
    """从文件名自动猜测 L/R，找不到就返回 default。"""
    name = os.path.basename(path).upper()
    if "_L_" in name or name.endswith("_L.NII") or name.endswith("_L.NII.GZ"):
        return 'L'
    if "_R_" in name or name.endswith("_R.NII") or name.endswith("_R.NII.GZ"):
        return 'R'
    return default


def _plane_basis_from_tibia(mask_bool, aff):
    """
    用胫骨区域估计“平台平面”及其二维坐标系：
    1) 世界坐标 PCA，取 PC3 近似 SI 轴
    2) 去掉沿 SI 的分量，在垂直平面再做 PCA 得到 (u, v)
    返回: 质心 c(3,), 平面基 (u(3,), v(3,))
    """
    idx = np.argwhere(mask_bool)
    if idx.size == 0:
        raise RuntimeError("Tibia reference mask is empty.")
    xyz = _vox2world(idx, aff)

    p_all = PCA(n_components=3).fit(xyz)
    si = p_all.components_[2] / np.linalg.norm(p_all.components_[2])  # 近似 SI 轴
    c = xyz.mean(axis=0)
    Xc = xyz - c
    Xp = Xc - (Xc @ si)[:, None] * si

    p_in = PCA(n_components=2).fit(Xp)
    u = p_in.components_[0] / np.linalg.norm(p_in.components_[0])
    v = p_in.components_[1] / np.linalg.norm(p_in.components_[1])
    return c, u, v


def _to_plane_uv(xyz, c, u, v):
    Xc = xyz - c
    return np.c_[Xc @ u, Xc @ v]  # (N,2)


def _kmeans_uv_split(ref_mask, aff, c, u, v, laterality):
    idx = np.argwhere(ref_mask)
    if idx.size == 0:
        raise RuntimeError("Reference mask empty in kmeans split.")
    xyz = _vox2world(idx, aff)
    uv  = _to_plane_uv(xyz, c, u, v)

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(uv)

    ctr_uv = km.cluster_centers_
    ctr_xyz = c + ctr_uv[:, 0][:, None] * u + ctr_uv[:, 1][:, None] * v

    x0, x1 = ctr_xyz[0, 0], ctr_xyz[1, 0]
    if laterality.upper() == 'L':
        lat_label = 0 if x0 > x1 else 1
    else:
        lat_label = 0 if x0 < x1 else 1
    med_label = 1 - lat_label
    return km, lat_label, med_label


def _split_any_mask(any_mask, aff, c, u, v, km, lat_label, med_label):
    if any_mask.sum() == 0:
        return any_mask.copy(), any_mask.copy()
    idx = np.argwhere(any_mask)
    xyz = _vox2world(idx, aff)
    uv  = _to_plane_uv(xyz, c, u, v)
    pred = km.predict(uv)
    lat_mask = np.zeros_like(any_mask, dtype=bool)
    med_mask = np.zeros_like(any_mask, dtype=bool)
    lat_mask[tuple(idx[pred == lat_label].T)] = True
    med_mask[tuple(idx[pred == med_label].T)] = True
    return lat_mask, med_mask


def split_compartments(in_multilabel, out_multilabel, laterality=None):
    nii = nib.load(in_multilabel)
    lab = nii.get_fdata().astype(np.int16)
    aff = nii.affine; hdr = nii.header

    if laterality is None or laterality.lower() == "auto":
        laterality = _auto_laterality_from_name(in_multilabel, default='L')

    # 参考区域优先顺序：胫骨软骨(21) > 胫骨下骨带(11)
    ref = (lab == CART_TIBIA)
    if ref.sum() < 200:
        ref = (lab == SUB_TIBIA)
    if ref.sum() < 200:
        raise RuntimeError("参考的胫骨平台区域太少（需要 CART_TIBIA 或 SUB_TIBIA）")

    c, u, v = _plane_basis_from_tibia(ref, aff)
    km, lat_label, med_label = _kmeans_uv_split(ref, aff, c, u, v, laterality)

    out = np.zeros_like(lab, dtype=np.int16)

    m = (lab == CART_FEMUR)
    if m.any():
        lat, med = _split_any_mask(m, aff, c, u, v, km, lat_label, med_label)
        out[lat] = LF; out[med] = MF

    m = (lab == CART_TIBIA)
    if m.any():
        lat, med = _split_any_mask(m, aff, c, u, v, km, lat_label, med_label)
        out[lat] = LT; out[med] = MT

    m = (lab == CART_PATELLA)
    if m.any():
        lat, med = _split_any_mask(m, aff, c, u, v, km, lat_label, med_label)
        out[lat] = LP; out[med] = MP

    m = (lab == SUB_FEMUR)
    if m.any():
        lat, med = _split_any_mask(m, aff, c, u, v, km, lat_label, med_label)
        out[lat] = LF_SUB; out[med] = MF_SUB

    m = (lab == SUB_TIBIA)
    if m.any():
        lat, med = _split_any_mask(m, aff, c, u, v, km, lat_label, med_label)
        out[lat] = LT_SUB; out[med] = MT_SUB

    nib.save(nib.Nifti1Image(out, aff, hdr), out_multilabel)
    print(f"[OK] {os.path.basename(in_multilabel)} -> {os.path.basename(out_multilabel)}")


def _default_outname(in_path):
    name = os.path.basename(in_path)
    if name.endswith("_all_rois_multilabel.nii.gz"):
        return name.replace("_all_rois_multilabel.nii.gz", "_all_lm_rois_multilabel.nii.gz")
    if name.endswith(".nii.gz"):
        return name[:-7] + "_lm.nii.gz"
    if name.endswith(".nii"):
        return name[:-4] + "_lm.nii.gz"
    return name + "_lm.nii.gz"


def split_folder(indir, outdir, pattern="*_all_rois_multilabel.nii.gz", lr="auto", stop_on_error=False):
    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(indir, pattern)))
    if not files:
        # 兜底：如果按默认 pattern 找不到，就改为 *.nii* 全部尝试
        files = sorted(glob.glob(os.path.join(indir, "*.nii*")))
    print(f"[INFO] Found {len(files)} files")

    n_ok, n_fail = 0, 0
    for f in files:
        try:
            out_name = _default_outname(f)
            out_path = os.path.join(outdir, out_name)
            split_compartments(f, out_path, laterality=lr)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[ERR] {os.path.basename(f)} failed: {e}")
            if stop_on_error:
                raise
    print(f"[DONE] ok={n_ok}, fail={n_fail}, outdir={outdir}")


def main():
    ap = argparse.ArgumentParser(description="Split knee compartments for an entire folder.")
    ap.add_argument("--indir", required=True, help="Input folder containing .nii/.nii.gz")
    ap.add_argument("--outdir", required=True, help="Output folder")
    ap.add_argument("--pattern", default="*_all_rois_multilabel.nii.gz", help="Glob pattern to match input files")
    ap.add_argument("--lr", default="auto", help="L/R side: L, R, or 'auto' (default)")
    ap.add_argument("--stop-on-error", action="store_true", help="Raise on first error")
    args = ap.parse_args()

    split_folder(args.indir, args.outdir, pattern=args.pattern, lr=args.lr, stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    main()
