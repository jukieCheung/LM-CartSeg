# LM-CartSeg

An automated knee MRI segmentation and radiomics analysis toolkit. It fuses dual-resolution (224 + 256) nnUNet segmentation outputs, automatically generates cartilage and subchondral bone band ROIs, splits them into lateral/medial compartments, and extracts morphological and radiomics features.

---

## Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | GUI / `knee_pipeline_studio_qt_v4.py` | Main Qt interface that orchestrates the entire pipeline |
| 2 | — | Resample to 256×256×160 |
| 3 | nnUNet | Dual-resolution (224 xyz + 256 zxy) segmentation inference |
| 4 | `clear_seg_224.py` / `clear_seg_256.py` | Post-processing (connected-component cleaning, label remapping) |
| 5 | `merged_clear.py` | Fuse 224 and 256 segmentation results |
| 6 | `batch_make_rois.py` | Extract cartilage and subchondral bone band ROIs |
| 7 | `split_lm_real1.py` | Split ROIs into lateral/medial (LM) compartments |
| 8 | `run_batch_cases.py` + `lm_knee_radiomics.py` | Batch radiomics feature extraction and morphological metrics |

---

## Input Orientation Convention

Before running the pipeline, all input knee MRI volumes should be standardized to a **right-knee orientation**. If the original scan is from a left knee, it should be mirrored to match the right-knee anatomical convention before segmentation, ROI generation, lateral/medial splitting, and radiomics extraction.

In this repository, the NIfTI array axes are interpreted as follows:

| NIfTI axis | Anatomical plane |
| ---------- | ---------------- |
| x          | coronal          |
| y          | axial            |
| z          | sagittal         |

Therefore, `xyz` in this pipeline refers to the NIfTI array-axis order:

```text
x = coronal
y = axial
z = sagittal
```

This convention is important because the downstream lateral/medial compartment splitting assumes a consistent left-right anatomical orientation across cases. Mixing native left-knee and right-knee scans without mirroring may lead to incorrect lateral/medial labels.

---

## File List

```
LM_CartSeg/
├── knee_pipeline_studio_qt_v4.py   # PySide6 GUI main program
├── batch_make_rois.py              # Batch ROI generation (cartilage + subchondral bone)
├── split_lm_real1.py               # Lateral/Medial compartment splitting
├── clear_seg_224.py                # Post-processing for 224-resolution predictions
├── clear_seg_256.py                # Post-processing for 256-resolution predictions
├── merged_clear.py                 # Fusion of 224/256 segmentation results
├── run_batch_cases.py              # Batch radiomics analysis entry point
├── lm_knee_radiomics.py            # Morphology metrics + PyRadiomics feature extraction
└── logo.png                        # GUI logo
```

---

## Dependencies

- Python >= 3.9
- [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet) (requires `nnUNetv2_predict` to be available)
- nibabel
- numpy
- scipy
- scikit-learn
- scikit-image
- pandas
- matplotlib
- [PyRadiomics](https://pypi.org/project/pyradiomics/)
- PySide6 (required only for the GUI)
- SimpleITK (optional, for high-quality resampling)

### Quick Install

```bash
pip install nibabel numpy scipy scikit-learn scikit-image pandas matplotlib pyradiomics pyside6 simpleitk
```

> nnUNetv2 should be installed separately following its official documentation, and environment variables (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) must be configured.

---

## Usage

### Method 1: GUI (Recommended)

```bash
python knee_pipeline_studio_qt_v4.py
```

Features:
- Select raw NIfTI folder and working directory
- Configure subchondral band thickness (mm) and radiomics bin width
- Toggle individual pipeline steps on/off
- Save/load presets (JSON), Dry-run mode
- Real-time log, progress bar, one-click open results folder

Settings are automatically saved to `~/.knee_pipeline_qt.json`.

### Method 2: Command Line (Step by Step)

#### 1. Post-process 224 predictions
```bash
python clear_seg_224.py --indir /path/to/pred224 --outdir /path/to/cleared224
```

#### 2. Post-process 256 predictions
```bash
python clear_seg_256.py --indir /path/to/pred256 --outdir /path/to/cleared256
```

#### 3. Fuse dual-resolution results
```bash
python merged_clear.py \
  --indir224 /path/to/cleared224 \
  --indir256 /path/to/cleared256 \
  --outdir /path/to/cleared_final
```

#### 4. Generate ROIs (cartilage + subchondral bone band)
```bash
python batch_make_rois.py \
  --indir /path/to/cleared_final \
  --outdir /path/to/rois \
  --band_mm 10
```

#### 5. Split into lateral/medial compartments
```bash
python split_lm_real1.py \
  --indir /path/to/rois \
  --outdir /path/to/lm_rois \
  --lr auto
```

#### 6. Batch radiomics analysis
```bash
python run_batch_cases.py \
  --images_root /path/to/resampled_images \
  --rois_root /path/to/lm_rois \
  --out_root /path/to/radiomics_output \
  --bin_width 5 \
  --heat_axis sag
```

---

## Label System

### Input Labels (nnUNet Raw Output)

| Value | Region |
|-------|--------|
| 1 | Femur Bone |
| 2 | Femur Cartilage |
| 3 | Tibia Bone |
| 4 | Tibia Cartilage |
| 5 | Patella Cartilage |

### Intermediate Labels (ROI Stage)

| Value | Region |
|-------|--------|
| 10 | Subchondral Femur |
| 11 | Subchondral Tibia |
| 20 | Femur Cartilage ROI |
| 21 | Tibia Cartilage ROI |
| 22 | Patella Cartilage ROI |

### Final Labels (After LM Splitting)

| Value | Region |
|-------|--------|
| 201 | Lateral Femur Cartilage |
| 202 | Medial Femur Cartilage |
| 203 | Lateral Tibia Cartilage |
| 204 | Medial Tibia Cartilage |
| 205 | Lateral Patella Cartilage |
| 206 | Medial Patella Cartilage |
| 211 | Lateral Femur Subchondral |
| 212 | Medial Femur Subchondral |
| 213 | Lateral Tibia Subchondral |
| 214 | Medial Tibia Subchondral |

---

## Output Files

For each case, `run_batch_cases.py` generates the following in `out_root/<case_id>/`:

| File | Description |
|------|-------------|
| `<case_id>_morphology.json` | Volume, mean thickness, and P95 thickness for each ROI |
| `<case_id>_radiomics.json` | First-order, texture, and shape features from PyRadiomics |
| `<case_id>_features.xlsx` / `.csv` | Combined table (one row per ROI, morphology + radiomics) |
| `<case_id>_<roi>_mask.nii.gz` | Individual ROI masks (intermediate files) |
| `t2_heatmap_<axis>.png` | T2 heatmap slice visualization |

---

## Lateral/Medial Splitting Algorithm (`split_lm_real1.py`)

1. Use tibial cartilage (or subchondral tibia) as the reference region.
2. Apply PCA to the world coordinates of the reference voxels to estimate the approximate SI axis.
3. Project onto the perpendicular plane and perform K-Means clustering (k=2) to obtain lateral and medial groups.
4. The algorithm assumes that all cases have been standardized to a right-knee orientation before splitting. Filename-based L/R inference is only retained for compatibility or manual checking.
5. 5. Apply the clustering model to all ROIs (cartilage + subchondral bone bands) to complete the lateral/medial split.

---

## Packaging as Executable

Use PyInstaller (run inside a virtual environment with all dependencies installed):

```bash
pyinstaller -F -n KneePipelineStudioQt --clean \
  --add-data "clear_seg_224.py;." \
  --add-data "clear_seg_256.py;." \
  --add-data "merged_clear.py;." \
  --add-data "batch_make_rois.py;." \
  --add-data "split_lm_real1.py;." \
  --add-data "run_batch_cases.py;." \
  --add-data "lm_knee_radiomics.py;." \
  --add-data "logo.png;." \
  knee_pipeline_studio_qt_v4.py
```

---

## License

- PySide6 is licensed under LGPL-3.0.
- The business logic code in this project is free to use.

---

## Citation

If you use this toolkit for academic research, please cite nnUNet and PyRadiomics:

- Isensee F, Jaeger P F, Kohl S A A, et al. **nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.** *Nature Methods*, 2021.
- van Griethuysen J J M, Fedorov A, Parmar C, et al. **Computational Radiomics System to Decode the Radiographic Phenotype.** *Cancer Research*, 2017.
- Zhang T, Li Z, Leung K L, FU S N. **LM-CartSeg: Automated Segmentation of Lateral and Medial Cartilage and Subchondral Bone for Radiomics Analysis.** *Arxiv*, 2026.
