#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, argparse
import numpy as np
import nibabel as nib

def merge_labels(dataA, dataB):
    merged = np.copy(dataA)
    maskA = dataA != 0
    maskB = dataB != 0
    merged[maskB & (~maskA)] = dataB[maskB & (~maskA)]
    conflict = maskA & maskB & (dataA != dataB)
    merged[conflict] = np.maximum(dataA[conflict], dataB[conflict])
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir224", required=True, help="Folder of cleaned 224 outputs")
    ap.add_argument("--indir256", required=True, help="Folder of cleaned 256 outputs")
    ap.add_argument("--outdir", required=True, help="Output folder for merged results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    nii256 = sorted([f for f in os.listdir(args.indir256) if f.endswith(".nii.gz")])
    nii224 = sorted([f for f in os.listdir(args.indir224) if f.endswith(".nii.gz")])

    name256 = {f: os.path.join(args.indir256, f) for f in nii256}
    name224 = {f: os.path.join(args.indir224, f) for f in nii224}
    common = sorted(set(name256) & set(name224))
    print(f"共找到 {len(common)} 个同名文件待合并")

    for name in common:
        pA = name256[name]; pB = name224[name]
        try:
            imgA = nib.load(pA); imgB = nib.load(pB)
            dataA = np.rint(imgA.get_fdata()).astype(np.int16)
            dataB = np.rint(imgB.get_fdata()).astype(np.int16)
            if dataA.shape != dataB.shape:
                print(f"❌ shape mismatch {name}: {dataA.shape}/{dataB.shape}")
                continue
            merged = merge_labels(dataA, dataB)
            out_path = os.path.join(args.outdir, name)
            nib.save(nib.Nifti1Image(merged, imgB.affine, imgB.header), out_path)
            print(f"[OK] merged -> {out_path}")
        except Exception as e:
            print(f"❌ Error on {name}: {e}")

    if not common:
        print("⚠️ No common filenames between 224/256 folders.")


if __name__ == "__main__":
    main()