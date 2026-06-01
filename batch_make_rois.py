#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch version: traverse a folder of label .nii.gz files and output *_all_rois_multilabel.nii.gz
"""

import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt

# Label IDs
BG = 0
FEMUR_BONE = 1
FEMUR_CART = 2
TIBIA_BONE = 3
TIBIA_CART = 4
PATELLA_CART = 5
PATELLA_BONE_LABEL = None  # set if available

# Output labels
SUB_FEMUR = 10
SUB_TIBIA = 11
SUB_PATELLA = 12
CART_FEMUR = 20
CART_TIBIA = 21
CART_PATELLA = 22


def load_nifti(path):
    nii = nib.load(path)
    return np.asanyarray(nii.dataobj), nii.affine, nii.header, np.array(nii.header.get_zooms()[:3])


def save_nifti(arr, affine, header_like, out_path, dtype=np.uint8):
    nib.save(nib.Nifti1Image(arr.astype(dtype), affine, header_like), out_path)


def make_subchondral_band(bone_mask, cartilage_mask, voxel_size_mm, band_mm):
    if not bone_mask.any() or not cartilage_mask.any():
        return np.zeros_like(bone_mask, dtype=bool)
    inv_cart = np.logical_not(cartilage_mask)
    dist_mm = distance_transform_edt(inv_cart, sampling=voxel_size_mm)
    return bone_mask & (dist_mm <= band_mm)


def process_file(label_path, out_dir, band_mm=10):
    lab, affine, hdr, vsz = load_nifti(label_path)

    femur_bone = lab == FEMUR_BONE
    femur_cart = lab == FEMUR_CART
    tibia_bone = lab == TIBIA_BONE
    tibia_cart = lab == TIBIA_CART
    patella_cart = lab == PATELLA_CART
    patella_bone = (lab == PATELLA_BONE_LABEL) if PATELLA_BONE_LABEL else np.zeros_like(lab, bool)

    sub_femur = make_subchondral_band(femur_bone, femur_cart, vsz, band_mm)
    sub_tibia = make_subchondral_band(tibia_bone, tibia_cart, vsz, band_mm)
    sub_patella = make_subchondral_band(patella_bone, patella_cart, vsz, band_mm) if PATELLA_BONE_LABEL else np.zeros_like(lab, bool)

    rois = np.zeros_like(lab, dtype=np.uint8)
    rois[sub_femur] = SUB_FEMUR
    rois[sub_tibia] = SUB_TIBIA
    if PATELLA_BONE_LABEL:
        rois[sub_patella] = SUB_PATELLA
    rois[femur_cart] = CART_FEMUR
    rois[tibia_cart] = CART_TIBIA
    rois[patella_cart] = CART_PATELLA

    fname = os.path.basename(label_path).replace(".nii.gz", "_all_rois_multilabel.nii.gz")
    out_path = os.path.join(out_dir, fname)
    save_nifti(rois, affine, hdr, out_path)
    print(f"[OK] {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Input folder containing .nii.gz label files")
    ap.add_argument("--outdir", required=True, help="Output folder for _all_rois_multilabel.nii.gz")
    ap.add_argument("--band_mm", type=float, default=10.0, help="Subchondral band thickness in mm")
    ap.add_argument("--patella_bone_label", type=int, default=-1, help="Label ID for patella bone if available")
    args = ap.parse_args()

    global PATELLA_BONE_LABEL
    PATELLA_BONE_LABEL = args.patella_bone_label if args.patella_bone_label >= 0 else None

    os.makedirs(args.outdir, exist_ok=True)

    nii_files = [f for f in os.listdir(args.indir) if f.endswith(".nii.gz")]
    if not nii_files:
        print("No .nii.gz files found in input folder.")
        return

    for f in nii_files:
        process_file(os.path.join(args.indir, f), args.outdir, band_mm=args.band_mm)

    print(f"[DONE] Processed {len(nii_files)} files. Results saved in: {args.outdir}")


if __name__ == "__main__":
    main()
