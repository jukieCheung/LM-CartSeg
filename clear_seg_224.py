#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import label as cc_label, generate_binary_structure

def clean_one_label(binary_mask,
                    keep_largest=True,
                    min_voxels=None,
                    min_volume_mm3=None,
                    voxel_spacing=None):
    if not binary_mask.any():
        return binary_mask
    structure = generate_binary_structure(3, 1)  # 3D 6-邻域
    labeled, num = cc_label(binary_mask, structure=structure)
    if num == 0:
        return np.zeros_like(binary_mask, dtype=bool)

    if keep_largest:
        sizes = np.bincount(labeled.ravel().astype(np.int64))
        sizes[0] = 0
        keep_id = sizes.argmax()
        return (labeled == keep_id)

    sizes = np.bincount(labeled.ravel().astype(np.int64))
    sizes[0] = 0
    keep_mask = np.zeros_like(binary_mask, dtype=bool)

    if min_voxels is None and (min_volume_mm3 is not None) and (voxel_spacing is not None):
        vx_vol = float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])  # mm^3
        min_voxels = int(np.floor(min_volume_mm3 / vx_vol))
    if min_voxels is None:
        min_voxels = 0

    for comp_id in range(1, len(sizes)):
        if sizes[comp_id] >= min_voxels:
            keep_mask |= (labeled == comp_id)
    return keep_mask


def clean_multilabel(seg_path,
                     out_path,
                     labels=None,
                     strategy_per_label=None,
                     default_keep_largest=True,
                     default_min_voxels=None,
                     default_min_volume_mm3=None):
    img = nib.load(seg_path)
    data = img.get_fdata().astype(np.int64)
    affine = img.affine
    header = img.header

    try:
        spacing = header.get_zooms()[:3]
    except Exception:
        spacing = None

    if labels is None:
        labels = [int(v) for v in np.unique(data) if v != 0]

    cleaned = np.zeros_like(data, dtype=np.int64)

    for lab in labels:
        mask = (data == lab)
        cfg = (strategy_per_label or {}).get(lab, {})
        keep_largest = cfg.get("keep_largest", default_keep_largest)
        min_voxels = cfg.get("min_voxels", default_min_voxels)
        min_volume_mm3 = cfg.get("min_volume_mm3", default_min_volume_mm3)

        cleaned_mask = clean_one_label(mask,
                                       keep_largest=keep_largest,
                                       min_voxels=min_voxels,
                                       min_volume_mm3=min_volume_mm3,
                                       voxel_spacing=spacing)
        cleaned[cleaned_mask] = lab

    nib.save(nib.Nifti1Image(cleaned.astype(np.int16), affine, header), out_path)
    print(f"Saved cleaned: {out_path}")


def relabel(data_int):
    # 删除 6、5
    data_int[data_int == 6] = 0
    data_int[data_int == 5] = 0
    # 1 -> 5
    data_int[data_int == 1] = 5
    # 3 -> 4（合并到4）
    data_int[data_int == 3] = 4
    return data_int


def process_one(nii_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img = nib.load(nii_path)
    data = img.get_fdata()
    data = np.rint(data).astype(np.int16)

    # 重映射
    data = relabel(data)

    # 先把重映射结果存到 out_dir 同名文件
    fname = os.path.basename(nii_path)
    out_relabeled = os.path.join(out_dir, fname)
    nib.save(nib.Nifti1Image(data, img.affine, img.header), out_relabeled)
    print(f"Relabeled -> {out_relabeled}")

    # 清理（用你给的策略），覆盖写回同名
    clean_multilabel(
        seg_path=out_relabeled,
        out_path=out_relabeled,
        strategy_per_label={
            2: {"keep_largest": True},
            4: {"keep_largest": False, "min_voxels": 500},
            5: {"keep_largest": True},
        },
        default_keep_largest=False,
        default_min_voxels=500
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Input folder containing .nii.gz (224 preds)")
    ap.add_argument("--outdir", required=True, help="Output folder for cleaned 224 files")
    args = ap.parse_args()

    nii_list = sorted([f for f in os.listdir(args.indir) if f.endswith(".nii.gz")])
    if not nii_list:
        print("No .nii.gz found in input folder.")
        return

    for f in nii_list:
        process_one(os.path.join(args.indir, f), args.outdir)

    print(f"[DONE] Cleaned {len(nii_list)} files -> {args.outdir}")


if __name__ == "__main__":
    main()