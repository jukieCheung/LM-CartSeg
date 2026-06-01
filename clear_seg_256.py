#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, glob, argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import label as cc_label, generate_binary_structure, center_of_mass

JSON_NAME = "dataset.json"

def clean_one_label(binary_mask,
                    keep_largest=True,
                    min_voxels=None,
                    min_volume_mm3=None,
                    voxel_spacing=None,
                    drop_border_touching=False,
                    prefer_center=False):
    if not binary_mask.any():
        return binary_mask
    structure = generate_binary_structure(3, 1)
    labeled, num = cc_label(binary_mask, structure=structure)
    if num == 0:
        return np.zeros_like(binary_mask, dtype=bool)

    sizes = np.bincount(labeled.ravel().astype(np.int64))
    sizes[0] = 0

    if min_voxels is None and (min_volume_mm3 is not None) and (voxel_spacing is not None):
        vx_vol = float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])
        min_voxels = int(np.floor(min_volume_mm3 / vx_vol))
    if min_voxels is None:
        min_voxels = 0

    valid = np.ones(num + 1, dtype=bool); valid[0] = False

    if drop_border_touching:
        border = np.zeros_like(labeled, dtype=bool)
        border[[0, -1], :, :] = True
        border[:, [0, -1], :] = True
        border[:, :, [0, -1]] = True
        border_ids = np.unique(labeled[border])
        valid[border_ids] = False

    small_ids = np.where(sizes < min_voxels)[0]
    valid[small_ids] = False

    candidates = np.where(valid)[0]
    if len(candidates) == 0:
        return np.zeros_like(binary_mask, dtype=bool)

    if keep_largest:
        best = candidates[np.argmax(sizes[candidates])]
        return (labeled == best)
    elif prefer_center:
        vol_center = np.array(binary_mask.shape) / 2.0
        dists = []
        for cid in candidates:
            com = np.array(center_of_mass(labeled == cid))
            dists.append(np.linalg.norm(com - vol_center))
        best = candidates[np.argmin(dists)]
        return (labeled == best)
    else:
        return np.isin(labeled, candidates)


def clean_multilabel(seg_path, out_path,
                     labels=None, strategy_per_label=None,
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
        cleaned_mask = clean_one_label(
            mask,
            keep_largest=cfg.get("keep_largest", default_keep_largest),
            min_voxels=cfg.get("min_voxels", default_min_voxels),
            min_volume_mm3=cfg.get("min_volume_mm3", default_min_volume_mm3),
            voxel_spacing=spacing,
            drop_border_touching=cfg.get("drop_border_touching", False),
            prefer_center=cfg.get("prefer_center", False),
        )
        cleaned[cleaned_mask] = lab

    nib.save(nib.Nifti1Image(cleaned.astype(np.int16), affine, header), out_path)
    print(f"Saved cleaned: {out_path}")


def merge_4_and_5_to_4(img):
    data = img.get_fdata()
    data[data == 5] = 4
    return data.astype(np.int16)


def process_one(p, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img = nib.load(p)
    merged = merge_4_and_5_to_4(img)
    fname = os.path.basename(p).replace(".nii", "").replace(".gz", "") + ".nii.gz"
    merged_path = os.path.join(out_dir, fname)
    nib.save(nib.Nifti1Image(merged, img.affine, img.header), merged_path)
    print(f"Merged 4&5->4: {merged_path}")

    clean_multilabel(
        seg_path=merged_path,
        out_path=merged_path,
        strategy_per_label={
            1: {"keep_largest": True, "prefer_center": True},
            2: {"keep_largest": False, "min_voxels": 2000},
            3: {"keep_largest": True, "prefer_center": True},
            4: {"keep_largest": False, "min_voxels": 1000},
        },
        default_keep_largest=False,
        default_min_voxels=200
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Input folder containing .nii.gz (256 preds)")
    ap.add_argument("--outdir", required=True, help="Output folder for cleaned 256 files")
    args = ap.parse_args()

    # Optional JSON update inside indir
    json_path = os.path.join(args.indir, JSON_NAME)
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            labels = dict(meta.get("labels", {}))
            new_labels = {}
            for k, v in labels.items():
                if v in (0, 1, 2, 3):
                    new_labels[k] = v
            new_labels["Tibial Cartilage"] = 4
            meta["labels"] = new_labels
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"✅ JSON updated: {json_path}")
        except Exception as e:
            print(f"⚠️ JSON update failed: {e}")

    nii_paths = sorted([f for f in os.listdir(args.indir) if f.endswith(".nii.gz")])
    if not nii_paths:
        print("No .nii.gz found in input folder.")
        return
    for f in nii_paths:
        process_one(os.path.join(args.indir, f), args.outdir)
    print(f"[DONE] Cleaned {len(nii_paths)} files -> {args.outdir}")


if __name__ == "__main__":
    main()
