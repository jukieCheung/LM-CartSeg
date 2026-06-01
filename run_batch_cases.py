# -*- coding: utf-8 -*-
"""
批量遍历运行 knee radiomics：
- 递归收集 images_root 下的所有 .nii / .nii.gz 影像
- 递归收集 rois_root 下的所有 *_all_rois_multilabel.nii.gz
- 自动配对：ROI 前缀（去掉 _all_rois_multilabel）与 image 文件名做匹配
- 每个配对在 out_root/<case_id>/ 下输出各自结果

用法示例：
python run_batch_cases.py \
  --images_root /path/to/images_root \
  --rois_root   /path/to/rois_root \
  --out_root    /path/to/output_root \
  --bin_width 5 --heat_axis sag

注意：
- ROI 文件名形如：<case_id>_all_rois_multilabel.nii.gz
- image 文件名建议包含同样的 <case_id>，例如 <case_id>.nii.gz 或 <case_id>_T2.nii.gz
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# 复用你已有的函数
from lm_knee_radiomics import run_case, save_t2_heatmap_png


def is_image(p: Path) -> bool:
    name = p.name.lower()
    return (name.endswith(".nii") or name.endswith(".nii.gz")) and ("_all_lm_rois_multilabel" not in name)


def is_roi(p: Path) -> bool:
    return p.name.lower().endswith("_all_lm_rois_multilabel.nii.gz")


def strip_roi_suffix(roi_name: str) -> str:
    # 去掉尾部后缀，得到 case_id
    # e.g. "MR_20191228_03_POLYU_005_LI_SIU_KOW__all_rois_multilabel.nii.gz" -> "MR_20191228_03_POLYU_005_LI_SIU_KOW_"
    # 若你的 ROI 没有多余下划线，直接替换即可：
    return roi_name.replace("_all_lm_rois_multilabel.nii.gz", "").replace("_ALL_LM_ROIS_MULTILABEL.nii.gz", "")


def collect_files(root: Path, predicate):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and predicate(p):
            files.append(p)
    return files


def build_image_index(image_paths):
    """
    建一个名字索引，方便匹配：
    - key: 纯文件名不含后缀（.nii / .nii.gz 全部去掉）
    - value: Path 列表（防止同名不同目录）
    另建一个“包含索引”用来做模糊匹配
    """
    index_exact = defaultdict(list)
    all_images = []

    for p in image_paths:
        stem = p.name
        if stem.lower().endswith(".nii.gz"):
            stem = stem[:-7]
        elif stem.lower().endswith(".nii"):
            stem = stem[:-4]
        index_exact[stem].append(p)
        all_images.append((stem, p))

    return index_exact, all_images


def find_image_for_roi(roi_path: Path, index_exact, all_images):
    """
    匹配策略：
    1) 精确匹配：用 roi 前缀与 image 的“纯文件名（去后缀）”完全相等
    2) 包含匹配：若找不到，找“image名包含roi前缀”或“roi前缀包含image名”的单一候选
    3) 多候选时优先最短编辑距离（简单用长度差 + 子串位置排序）
    """
    roi_prefix = strip_roi_suffix(roi_path.name)
    roi_prefix_noext = roi_prefix  # 已经不含后缀了

    # 精确匹配
    if roi_prefix_noext in index_exact:
        # 若多个同名，选路径最短（通常更靠近）
        cands = sorted(index_exact[roi_prefix_noext], key=lambda p: len(str(p)))
        return cands[0], roi_prefix_noext

    # 包含匹配
    contain_cands = []
    for img_stem, img_path in all_images:
        if roi_prefix_noext in img_stem or img_stem in roi_prefix_noext:
            # 简单打分：长度差 + 在字符串中的位置（越小越好）
            if roi_prefix_noext in img_stem:
                pos = img_stem.find(roi_prefix_noext)
            else:
                pos = roi_prefix_noext.find(img_stem)
            score = abs(len(img_stem) - len(roi_prefix_noext)) + max(pos, 0)
            contain_cands.append((score, img_path, img_stem))

    if contain_cands:
        contain_cands.sort(key=lambda x: (x[0], len(str(x[1]))))
        best = contain_cands[0]
        return best[1], best[2]

    return None, roi_prefix_noext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True, help="原始影像根目录（递归扫描 .nii/.nii.gz）")
    ap.add_argument("--rois_root",   required=True, help="ROI 根目录（递归扫描 *_all_rois_multilabel.nii.gz）")
    ap.add_argument("--out_root",    required=True, help="输出根目录（每例一个子文件夹）")
    ap.add_argument("--bin_width", type=float, default=5.0, help="PyRadiomics binWidth，默认 5")
    ap.add_argument("--heat_axis", type=str, default="sag", choices=["sag","cor","axi"], help="热力图切面")
    ap.add_argument("--heat_index", type=int, default=None, help="热力图切片索引（默认居中）")
    ap.add_argument("--dry_run", action="store_true", help="仅打印配对，不实际运行")
    args = ap.parse_args()

    images_root = Path(args.images_root).expanduser().resolve()
    rois_root   = Path(args.rois_root).expanduser().resolve()
    out_root    = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    image_paths = collect_files(images_root, is_image)
    roi_paths   = collect_files(rois_root,   is_roi)

    if not image_paths:
        print(f"[ERROR] 未在 {images_root} 发现任何 .nii/.nii.gz 影像")
        sys.exit(1)
    if not roi_paths:
        print(f"[ERROR] 未在 {rois_root} 发现任何 *_all_rois_multilabel.nii.gz")
        sys.exit(1)

    index_exact, all_images = build_image_index(image_paths)

    print(f"[INFO] 收集到 image: {len(image_paths)} 个, ROI: {len(roi_paths)} 个")
    paired = []
    unmatched = []

    for roi in sorted(roi_paths):
        img, case_id = find_image_for_roi(roi, index_exact, all_images)
        if img is None:
            unmatched.append(roi)
            print(f"[WARN] ROI 无匹配 image: {roi}")
            continue
        paired.append((case_id, img, roi))

    print(f"\n[SUMMARY] 成功配对 {len(paired)} 对；未匹配 ROI {len(unmatched)} 个。")
    if unmatched:
        print("未匹配 ROI 示例：")
        for x in unmatched[:10]:
            print("  -", x)

    if args.dry_run:
        print("\n[DRY-RUN] 仅展示前 10 对匹配：")
        for case_id, img, roi in paired[:10]:
            print(f"  case_id={case_id}\n    image={img}\n    roi  ={roi}")
        return

    # 正式跑
    for case_id, img_path, roi_path in paired:
        case_out = out_root / case_id
        case_out.mkdir(parents=True, exist_ok=True)

        try:
            # 1) 形态学 + radiomics
            metrics, feats = run_case(
                image_path = str(img_path),
                total_roi_path = str(roi_path),
                out_dir = str(case_out),
                id_tag = case_id,
                bin_width = args.bin_width
            )

            # 2) 热力图
            png_path = case_out / f"t2_heatmap_{args.heat_axis}.png"
            save_t2_heatmap_png(
                image_path = str(img_path),
                total_roi_path = str(roi_path),
                out_png = str(png_path),
                axis = args.heat_axis if args.heat_axis!="axi" else "axi",
                idx = args.heat_index
            )

            print(f"[DONE] {case_id} -> {case_out}")
        except Exception as e:
            print(f"[ERROR] {case_id} 失败：{e}")

if __name__ == "__main__":
    main()


