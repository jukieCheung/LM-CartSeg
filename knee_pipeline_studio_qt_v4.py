#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knee Pipeline Studio — Qt 版 (PySide6 + MIT/QFluent 风格可选)
- 目标：去掉 PySimpleGUI 依赖，换 LGPL 的 PySide6（Qt 官方 Python 绑定）。
- 许可：PySide6 = LGPL-3.0，可商用（需动态链接 Qt 库、保留许可声明、允许用户替换库）。
- 可选美化：QFluentWidgets (MIT)；此示例先用原生 Qt 控件，方便落地。

功能对齐：
• 左右双栏：参数区 / 日志区
• 进度条 + 当前步骤
• Start / Stop / Dry-run
• 预设保存/加载（JSON）+ 自动记忆
• 依赖/脚本存在性检查
• 打开工作目录 / 结果目录
• 子进程实时日志（subprocess + 线程）

打包：PyInstaller 可行（示例见文末注释）。
"""
from __future__ import annotations
import sys, os, json, time, threading, queue, subprocess, shutil, glob
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

try:
    import nibabel as nib
    import numpy as np
except Exception:
    nib = None
    np = None

try:
    import SimpleITK as sitk
    HAS_SITK = True
except Exception:
    HAS_SITK = False

APP_TITLE = "Knee Pipeline Studio"
APP_VER = "v1"
STEPS = [
    "Resample to 256×256×160",
    "nnUNet segmentation",
    "Clean segmentation",
    "Extract cartilage and subchondral Bone",
    "Split lateral and medial",
    "Radiomics (LM rois)",      # <<< NEW
]


class Worker(QtCore.QThread):
    logLine = QtCore.Signal(str)
    stepChanged = QtCore.Signal(int)
    success = QtCore.Signal(float)
    failed = QtCore.Signal(str)

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    # ------------- helpers -------------
    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.logLine.emit(f"[{ts}] {msg}")

    def run_cmd(self, cmd, check=True, env=None):
        self.log(f"$ {' '.join(map(str, cmd))}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            if self._stop.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise RuntimeError("Stopped by user")
            self.log(line.rstrip())
        ret = proc.wait()
        if check and ret != 0:
            raise RuntimeError(f"Command failed (exit {ret}): {' '.join(map(str, cmd))}")
        return ret

    def which(self, c: str) -> Optional[str]:
        return shutil.which(c)

    # ------------- resample -------------
    def sitk_resample_to_256(self, in_path, out_path, target_shape=(256,256,160)):
        img = sitk.ReadImage(str(in_path))
        size = img.GetSize()  # x,y,z
        spacing = img.GetSpacing()
        scale = [size[i]/target_shape[i] for i in range(3)]
        new_spacing = (spacing[0]*scale[0], spacing[1]*scale[1], spacing[2]*scale[2])
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([int(v) for v in target_shape])
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetInterpolator(sitk.sitkBSpline)
        out_img = resampler.Execute(img)
        sitk.WriteImage(out_img, str(out_path))

    def nib_resample_nearest(self, in_path, out_path, target_shape=(256,256,160)):
        if nib is None or np is None:
            raise RuntimeError("nibabel/numpy not available for fallback resample")
        nii = nib.load(str(in_path))
        data = nii.get_fdata()
        sx, sy, sz = target_shape
        rx = max(1, round(sx / data.shape[0]))
        ry = max(1, round(sy / data.shape[1]))
        rz = max(1, round(sz / data.shape[2]))
        arr = np.repeat(np.repeat(np.repeat(data, rx, axis=0), ry, axis=1), rz, axis=2)
        arr = arr[:sx, :sy, :sz]
        nib.save(nib.Nifti1Image(arr.astype(nii.get_data_dtype()), nii.affine, nii.header), str(out_path))

    def resample_all(self, in_dir, out_dir):
        files = sorted(glob.glob(str(Path(in_dir)/"*.nii*")))
        total = len(files)
        for i, f in enumerate(files, 1):
            if self._stop.is_set():
                raise RuntimeError("Stopped by user")
            dst = Path(out_dir)/ (Path(f).stem.replace('.nii','') + '.nii.gz')
            if dst.exists():
                self.log(f"[SKIP {i}/{total}] exists: {dst.name}")
                continue
            try:
                if HAS_SITK:
                    self.sitk_resample_to_256(f, dst)
                else:
                    self.nib_resample_nearest(f, dst)
                self.log(f"[OK {i}/{total}] resampled → {dst.name}")
            except Exception as e:
                self.log(f"[ERR {i}/{total}] {e}")
                raise

    # ------------- permutations -------------
    def permute_xyz_to_zxy(self, img, outp):
        if nib is None or np is None:
            raise RuntimeError("nibabel/numpy required for axis permute")
        nii = nib.load(str(img))
        arr = nii.get_fdata()
        arr_perm = np.transpose(arr,(2,0,1))
        nib.save(nib.Nifti1Image(arr_perm.astype(nii.get_data_dtype()), nii.affine, nii.header), str(outp))

    def permute_zxy_to_xyz(self, img, outp):
        if nib is None or np is None:
            raise RuntimeError("nibabel/numpy required for axis permute")
        nii = nib.load(str(img))
        arr = nii.get_fdata()
        arr_perm = np.transpose(arr,(1,2,0))
        nib.save(nib.Nifti1Image(arr_perm.astype(nii.get_data_dtype()), nii.affine, nii.header), str(outp))

    def batch_make_zxy(self, in_dir, out_dir):
        files = sorted(glob.glob(str(Path(in_dir)/"*.nii*")))
        for f in files:
            self.permute_xyz_to_zxy(f, Path(out_dir)/Path(f).name)

    def batch_restore_xyz_from_pred256(self, pred_dir):
        files = sorted(glob.glob(str(Path(pred_dir)/"*nii*")))
        for f in files:
            if f.endswith("_xyz.nii.gz"):  # already restored
                continue
            dst = Path(f).with_name(Path(f).stem.replace('.nii','') + '.nii.gz')
            self.permute_zxy_to_xyz(f, dst)

    # ------------- nnUNet / external steps -------------
    def run_nnunet(self, dataset_spec, in_dir, out_dir, folds="0", cfg="3d_fullres"):
        cmd = [
            "nnUNetv2_predict","-d",str(dataset_spec),"-i",str(in_dir),"-o",str(out_dir),"-c",cfg,"-f",str(folds),"--disable_tta"
        ]
        self.run_cmd(cmd)

    def clear_seg_224(self, gui_dir, indir, outdir):
        self.run_cmd([sys.executable, str(Path(gui_dir)/"clear_seg_224.py"), "--indir", str(indir), "--outdir", str(outdir)])

    def clear_seg_256(self, gui_dir, indir, outdir):
        self.run_cmd([sys.executable, str(Path(gui_dir)/"clear_seg_256.py"), "--indir", str(indir), "--outdir", str(outdir)])

    def merged_clear(self, gui_dir, indir1, indir2, outdir):
        self.run_cmd([sys.executable, str(Path(gui_dir)/"merged_clear.py"), "--indir224", str(indir1), "--indir256", str(indir2), "--outdir", str(outdir)])

    def batch_make_rois(self, gui_dir, indir, outdir, band_mm):
        self.run_cmd([sys.executable, str(Path(gui_dir)/"batch_make_rois.py"), "--indir", str(indir), "--outdir", str(outdir), "--band_mm", str(band_mm)])

    def split_lm(self, gui_dir, indir, outdir):
        self.run_cmd([sys.executable, str(Path(gui_dir)/"split_lm_real1.py"), "--indir", str(indir), "--outdir", str(outdir)])

    def run_radiomics(self, gui_dir, images_root, rois_root, out_root,
                      bin_width=25, heat_axis="sag", heat_index=None):
        """
        调用 run_batch_cases.py 批量跑 radiomics
        images_root: 一般用 resampled 目录
        rois_root:   lm_rois 目录（含 *_all_lm_rois_multilabel.nii.gz）
        out_root:    radiomics 输出目录
        """
        cmd = [
            sys.executable, str(Path(gui_dir) / "run_batch_cases.py"),
            "--images_root", str(images_root),
            "--rois_root",   str(rois_root),
            "--out_root",    str(out_root),
            "--bin_width",   str(bin_width),
            "--heat_axis",   str(heat_axis),
        ]
        if heat_index is not None:
            cmd += ["--heat_index", str(heat_index)]
        self.run_cmd(cmd)



    # ------------- lifecycle -------------
    def run(self):
        start = time.time()
        try:
            cfg = self.cfg
            steps = cfg.get('steps', [True]*len(STEPS))
            raw_dir = Path(cfg["raw_dir"]).resolve()
            work_dir = Path(cfg["work_dir"]).resolve()
            gui_dir  = Path(cfg["gui_dir"]).resolve()

            res_dir = work_dir/"resampled"
            zxy_dir = work_dir/"zxy_for_256"
            pred224_dir = work_dir/"pred_224"
            pred256_dir = work_dir/"pred_256"
            cleared224 = work_dir/"cleared_224"
            cleared256 = work_dir/"cleared_256"
            cleared_final = work_dir/"cleared_final"
            rois_dir = work_dir/"rois"
            lm_dir   = work_dir/"lm_rois"
            rad_dir  = work_dir/"radiomics"   # <<< NEW


            for d in [work_dir,res_dir,zxy_dir,pred224_dir,pred256_dir,cleared224,cleared256,cleared_final,rois_dir,lm_dir]:
                d.mkdir(parents=True, exist_ok=True)

            self.log(f"[Paths]\n  gui_dir={gui_dir}\n  work_dir={work_dir}\n  raw_dir={raw_dir}\n")

            # 0 resample
            if steps[0]:
                self.stepChanged.emit(0)
                self.log("====== [Step 0] Resample ======")
                self.resample_all(raw_dir, res_dir)
            else:
                self.stepChanged.emit(0)
                self.log("[SKIP] Step 0 Resample")

            # 1 nnUNet predictions (224 + 256)
            if steps[1]:
                self.stepChanged.emit(1)
                self.log("====== [Step 1] nnUNet: 224 (xyz) ======")
                if not cfg.get("dry_run"):
                    self.run_nnunet(cfg["dataset224"], res_dir, pred224_dir, folds=cfg["folds"], cfg=cfg["nnunet_cfg"])
                else:
                    time.sleep(0.2); self.log("[DRY] nnUNet 224 skipped…")

                self.log("====== [Step 1] nnUNet: 256 (zxy) + restore ======")
                self.batch_make_zxy(res_dir, zxy_dir)
                if not cfg.get("dry_run"):
                    self.run_nnunet(cfg["dataset256"], zxy_dir, pred256_dir, folds=cfg["folds"], cfg=cfg["nnunet_cfg"])
                    self.batch_restore_xyz_from_pred256(pred256_dir)
                else:
                    time.sleep(0.2); self.log("[DRY] nnUNet 256 skipped…")
            else:
                self.stepChanged.emit(1)
                self.log("[SKIP] Step 1 nnUNet predictions (224+256)")

            # 2 Clean & Merge
            if steps[2]:
                self.stepChanged.emit(2)
                self.log("====== [Step 2] clear_seg_224.py ======")
                if not cfg.get("dry_run"): self.clear_seg_224(gui_dir, pred224_dir, cleared224)
                else: time.sleep(0.1); self.log("[DRY] clear 224 skipped…")

                self.log("====== [Step 2] clear_seg_256.py ======")
                if not cfg.get("dry_run"): self.clear_seg_256(gui_dir, pred256_dir, cleared256)
                else: time.sleep(0.1); self.log("[DRY] clear 256 skipped…")

                self.log("====== [Step 2] merged_clear.py ======")
                if not cfg.get("dry_run"): self.merged_clear(gui_dir, cleared224, cleared256, cleared_final)
                else: time.sleep(0.1); self.log("[DRY] merge skipped…")
            else:
                self.stepChanged.emit(2)
                self.log("[SKIP] Step 2 Clean & Merge")

            # 3 ROIs
            if steps[3]:
                self.stepChanged.emit(3)
                self.log("====== [Step 3] batch_make_rois.py ======")
                if not cfg.get("dry_run"): self.batch_make_rois(gui_dir, cleared_final, rois_dir, cfg["band_mm"])
                else: time.sleep(0.1); self.log("[DRY] rois skipped…")
            else:
                self.stepChanged.emit(3)
                self.log("[SKIP] Step 3 make ROIs")

            # 4 split LM
            if steps[4]:
                self.stepChanged.emit(4)
                self.log("====== [Step 4] split_lm_real1.py ======")
                if not cfg.get("dry_run"):
                    self.split_lm(gui_dir, rois_dir, lm_dir)
                else:
                    time.sleep(0.1); self.log("[DRY] split lm skipped…")
            else:
                self.stepChanged.emit(4)
                self.log("[SKIP] Step 4 split LM")

            # 5 Radiomics
            if len(steps) > 5 and steps[5]:
                self.stepChanged.emit(5)
                self.log("====== [Step 5] Radiomics (run_batch_cases.py) ======")
                if not cfg.get("dry_run"):
                    self.run_radiomics(
                        gui_dir=gui_dir,
                        images_root=res_dir,            # 用 resampled 图像
                        rois_root=lm_dir,              # LM ROI
                        out_root=rad_dir,
                        bin_width=cfg.get("rad_bin_width", 5.0),
                        heat_axis=cfg.get("rad_heat_axis", "sag"),
                        heat_index=cfg.get("rad_heat_index", None),
                    )
                else:
                    time.sleep(0.1); self.log("[DRY] radiomics skipped…")
            else:
                if len(steps) > 5:
                    self.stepChanged.emit(5)
                self.log("[SKIP] Step 5 Radiomics")

            self.success.emit(time.time()-start)

        except Exception as e:
            self.failed.emit(str(e))

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1060, 640)

        self.settings_path = Path.home()/".knee_pipeline_qt.json"
        self.worker: Optional[Worker] = None

        # ---- UI ----
        cw = QtWidgets.QWidget(self)
        self.setCentralWidget(cw)
        h = QtWidgets.QHBoxLayout(cw)

        # Left controls
        left = QtWidgets.QVBoxLayout()
        # Header with optional logo + title
        header_row = QtWidgets.QHBoxLayout()
        self.logo_label = QtWidgets.QLabel()
        try:
            logo_path = Path(__file__).with_name("logo.png")
            if logo_path.exists():
                pm = QtGui.QPixmap(str(logo_path)).scaledToHeight(28, QtCore.Qt.SmoothTransformation)
                self.logo_label.setPixmap(pm)
                self.setWindowIcon(QtGui.QIcon(pm))
        except Exception:
            pass
        title_lbl = QtWidgets.QLabel("LM-CartSeg")
        title_lbl.setStyleSheet("font-weight:600; font-size:16px;")
        header_row.addWidget(self.logo_label)
        header_row.addWidget(title_lbl)
        header_row.addStretch(1)
        left.addLayout(header_row)
        self.raw_edit = self._file_row(left, "Raw NIfTI folder", folder=True)
        self.work_edit = self._file_row(left, "Work/output folder", folder=True)

        grid = QtWidgets.QGridLayout()
        left.addLayout(grid)
        '''
        grid.addWidget(QtWidgets.QLabel("Dataset 224"), 0, 0)
        self.d224 = QtWidgets.QLineEdit("224")
        grid.addWidget(self.d224, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Dataset 256"), 0, 2)
        self.d256 = QtWidgets.QLineEdit("256")
        grid.addWidget(self.d256, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Folds"), 1, 0)
        self.folds = QtWidgets.QLineEdit("0")
        self.folds.setMaximumWidth(80)
        grid.addWidget(self.folds, 1, 1)
        grid.addWidget(QtWidgets.QLabel("nnUNet cfg"), 1, 2)
        self.cfg = QtWidgets.QLineEdit("3d_fullres")
        grid.addWidget(self.cfg, 1, 3)
        '''
        grid.addWidget(QtWidgets.QLabel("Band (mm)"), 2, 0)
        self.band = QtWidgets.QLineEdit("10")
        self.band.setMaximumWidth(80)
        grid.addWidget(self.band, 2, 1)

        # --- Radiomics 参数 (NEW) ---
        grid.addWidget(QtWidgets.QLabel("Bin width"), 3, 0)
        self.binwidth = QtWidgets.QLineEdit("5")
        self.binwidth.setMaximumWidth(80)
        grid.addWidget(self.binwidth, 3, 1)

        grid.addWidget(QtWidgets.QLabel("Heat axis"), 3, 2)
        self.heat_axis = QtWidgets.QComboBox()
        self.heat_axis.addItems(["sag", "cor", "axi"])
        grid.addWidget(self.heat_axis, 3, 3)

        self.dry = QtWidgets.QCheckBox("Dry-run (skip external cmds)")
        left.addWidget(self.dry)

        # Toggle to show/hide log pane
        self.show_log = QtWidgets.QCheckBox("Show log panel")
        self.show_log.setChecked(False)
        left.addWidget(self.show_log)

        # Steps group with per-step toggles
        steps_group = QtWidgets.QGroupBox("Steps to run")
        steps_layout = QtWidgets.QGridLayout(steps_group)
        self.step_checks = []
        for i, name in enumerate(STEPS):
            cb = QtWidgets.QCheckBox(f"{i}. {name}")
            cb.setChecked(True)
            self.step_checks.append(cb)
            r, c = divmod(i, 2)
            steps_layout.addWidget(cb, r, c)
        row_btns = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton("Select All")
        self.btn_none = QtWidgets.QPushButton("Select None")
        row_btns.addStretch(1)
        row_btns.addWidget(self.btn_all)
        row_btns.addWidget(self.btn_none)
        steps_layout.addLayout(row_btns, (len(STEPS)+1)//2, 0, 1, 2)
        left.addWidget(steps_group)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("▶ Start")
        self.stop_btn = QtWidgets.QPushButton("■ Stop")
        self.save_preset_btn = QtWidgets.QPushButton("Save Preset")
        self.load_preset_btn = QtWidgets.QPushButton("Load Preset")
        self.open_work_btn = QtWidgets.QPushButton("Open Work Dir")
        self.open_res_btn  = QtWidgets.QPushButton("Open Results")
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.save_preset_btn)
        btn_row.addWidget(self.load_preset_btn)
        btn_row.addWidget(self.open_work_btn)
        btn_row.addWidget(self.open_res_btn)
        left.addLayout(btn_row)

        # Status
        self.status_lbl = QtWidgets.QLabel("Ready")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, len(STEPS))
        left.addWidget(self.status_lbl)
        left.addWidget(self.progress)
        left.addStretch(1)

        # Right log
        right = QtWidgets.QVBoxLayout()
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        font = QtGui.QFont("Consolas", 10)
        self.log.setFont(font)
        right.addWidget(self.log)

        # Wrap right panel so we can hide/show logs easily
        self.rightWidget = QtWidgets.QWidget()
        self.rightWidget.setLayout(right)

        h.addLayout(left, 0)
        h.addWidget(self._separator(), 0)
        h.addWidget(self.rightWidget, 1)

        # wires
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.save_preset_btn.clicked.connect(self.on_save_preset)
        self.load_preset_btn.clicked.connect(self.on_load_preset)
        self.open_work_btn.clicked.connect(self.on_open_work)
        self.open_res_btn.clicked.connect(self.on_open_res)
        self.show_log.toggled.connect(self.rightWidget.setVisible)
        self.btn_all.clicked.connect(lambda: [cb.setChecked(True) for cb in self.step_checks])
        self.btn_none.clicked.connect(lambda: [cb.setChecked(False) for cb in self.step_checks])
        # default hide log panel
        self.rightWidget.setVisible(False)

        self.restore_state()

    # --- helpers ---
    def _separator(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    def _file_row(self, layout, label, folder=False):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        edit = QtWidgets.QLineEdit()
        btn = QtWidgets.QPushButton("Browse…")
        def pick():
            if folder:
                d = QtWidgets.QFileDialog.getExistingDirectory(self, label)
                if d:
                    edit.setText(d)
            else:
                f, _ = QtWidgets.QFileDialog.getOpenFileName(self, label)
                if f:
                    edit.setText(f)
        btn.clicked.connect(pick)
        row.addWidget(edit)
        row.addWidget(btn)
        layout.addLayout(row)
        return edit

    def append_log(self, text: str):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def toast(self, text: str):
        QtWidgets.QMessageBox.information(self, APP_TITLE, text)

    def error(self, text: str):
        QtWidgets.QMessageBox.critical(self, APP_TITLE, text)

    # --- state ---
    def restore_state(self):
        if self.settings_path.exists():
            try:
                data = json.loads(self.settings_path.read_text(encoding='utf-8'))
                self.raw_edit.setText(data.get('raw',''))
                self.work_edit.setText(data.get('work',''))
                #self.d224.setText(data.get('d224','224'))
                #self.d256.setText(data.get('d256','256'))
                #self.folds.setText(data.get('folds','0'))
                #self.cfg.setText(data.get('cfg','3d_fullres'))
                self.band.setText(data.get('band','10'))
                self.binwidth.setText(data.get('binwidth','5'))  # NEW
                self.dry.setChecked(bool(data.get('dry', False)))

                ha = data.get('heat_axis', 'sag')                 # NEW
                idx = self.heat_axis.findText(ha)
                if idx >= 0:
                    self.heat_axis.setCurrentIndex(idx)

                steps = data.get('steps')
                if isinstance(steps, list):
                    for cb, v in zip(self.step_checks, steps):
                        cb.setChecked(bool(v))
            except Exception:
                pass

    def save_state(self):
        data = dict(
            raw=self.raw_edit.text(),
            work=self.work_edit.text(),
            #d224=self.d224.text(),
            #d256=self.d256.text(),
            #folds=self.folds.text(),
            #fg=self.cfg.text(),
            band=self.band.text(),
            binwidth=self.binwidth.text(),                  # NEW
            heat_axis=self.heat_axis.currentText(),         # NEW
            dry=self.dry.isChecked(),
            steps=[cb.isChecked() for cb in self.step_checks]
        )

        try:
            self.settings_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception:
            pass

    # --- actions ---
    def on_start(self):
        raw = self.raw_edit.text().strip()
        work = self.work_edit.text().strip()
        if not raw or not os.path.isdir(raw):
            self.error("Please select a valid Raw folder")
            return
        if not work:
            self.error("Please choose a Work/output folder")
            return
        # prereqs
        gui_dir = Path(__file__).resolve().parent
        missing = []
        if shutil.which("nnUNetv2_predict") is None and not self.dry.isChecked():
            missing.append("nnUNetv2_predict")
        for s in [
            "clear_seg_224.py",
            "clear_seg_256.py",
            "merged_clear.py",
            "batch_make_rois.py",
            "split_lm_real1.py",
            "run_batch_cases.py",          # NEW
        ]:
            if not (gui_dir/s).exists():
                missing.append(s)

        if missing and not self.dry.isChecked():
            self.error("Missing prerequisites:\n- " + "\n- ".join(missing))
            return

        try:
            band = float(self.band.text())
        except Exception:
            self.error("Band (mm) must be a number")
            return

        try:
            binwidth = float(self.binwidth.text())
        except Exception:
            self.error("Bin width must be a number")
            return

        cfg = dict(
            raw_dir=raw,
            work_dir=work,
            gui_dir=str(gui_dir),
            #dataset224=self.d224.text().strip(),
            #dataset256=self.d256.text().strip(),
            #folds=self.folds.text().strip(),
            #nnunet_cfg=self.cfg.text().strip(),
            dataset224="224",
            dataset256="256",
            folds="0",
            nnunet_cfg="3d_fullres",
            band_mm=band,
            dry_run=self.dry.isChecked(),
            steps=[cb.isChecked() for cb in self.step_checks],
            # radiomics 相关配置
            rad_bin_width=binwidth,
            rad_heat_axis=self.heat_axis.currentText().strip(),
            rad_heat_index=None,   # 先用自动选 slice，有需要你再加 UI
        )

        self.save_state()

        self.log.setPlainText("")
        self.append_log(f"Starting pipeline ({APP_VER})\n")
        self.status_lbl.setText("Running…")
        self.progress.setValue(0)
        self.set_controls_enabled(False)

        self.worker = Worker(cfg)
        self.worker.logLine.connect(self.append_log)
        self.worker.stepChanged.connect(self.on_step)
        self.worker.success.connect(self.on_success)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def set_controls_enabled(self, enabled: bool):
        widgets = [
            self.raw_edit, self.work_edit,
            #self.d224, self.d256,
            #self.folds, self.cfg,
            self.band,
            self.binwidth, self.heat_axis,   # NEW
            self.save_preset_btn, self.load_preset_btn,
            *self.step_checks
        ]
        for w in widgets:
            w.setEnabled(enabled)


    def on_stop(self):
        if self.worker and self.worker.isRunning():
            ret = QtWidgets.QMessageBox.question(self, APP_TITLE, "Stop the running pipeline?")
            if ret == QtWidgets.QMessageBox.Yes:
                self.worker.stop()

    def on_step(self, idx: int):
        self.progress.setValue(idx)
        if 0 <= idx < len(STEPS):
            self.status_lbl.setText(f"Running: {STEPS[idx]}")

    def on_success(self, s: float):
        self.append_log(f"\n✅ SUCCESS in {s:.1f}s")
        self.status_lbl.setText("Done")
        self.progress.setValue(len(STEPS))
        self.set_controls_enabled(True)

    def on_failed(self, err: str):
        self.append_log("\n❌ " + err)
        self.status_lbl.setText("Error")
        self.set_controls_enabled(True)

    def on_save_preset(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save preset", filter="JSON (*.json)")
        if not fn: return
        data = dict(
            raw=self.raw_edit.text(), work=self.work_edit.text(),
            #d224=self.d224.text(), d256=self.d256.text(),
            #folds=self.folds.text(), cfg=self.cfg.text(),
            band=self.band.text(),
            binwidth=self.binwidth.text(),                    # NEW
            heat_axis=self.heat_axis.currentText(),           # NEW
            dry=self.dry.isChecked(),
            steps=[cb.isChecked() for cb in self.step_checks]
        )

        Path(fn).write_text(json.dumps(data, indent=2), encoding='utf-8')
        self.toast("Preset saved")

    def on_load_preset(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load preset", filter="JSON (*.json)")
        if not fn: return
        data = json.loads(Path(fn).read_text(encoding='utf-8'))
        self.raw_edit.setText(data.get('raw',''))
        self.work_edit.setText(data.get('work',''))
        self.d224.setText(data.get('d224','224'))
        self.d256.setText(data.get('d256','256'))
        self.folds.setText(data.get('folds','0'))
        self.cfg.setText(data.get('cfg','3d_fullres'))
        self.band.setText(data.get('band','10'))
        self.binwidth.setText(data.get('binwidth','5'))       # NEW
        self.dry.setChecked(bool(data.get('dry', False)))
        ha = data.get('heat_axis','sag')                       # NEW
        idx = self.heat_axis.findText(ha)
        if idx >= 0:
            self.heat_axis.setCurrentIndex(idx)
        steps = data.get('steps')
        if isinstance(steps, list):
            for cb, v in zip(self.step_checks, steps):
                cb.setChecked(bool(v))
        self.toast("Preset loaded")

    def on_open_work(self):
        p = self.work_edit.text().strip()
        if not p: return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(p))

    def on_open_res(self):
        p = self.work_edit.text().strip()
        if not p: return
        res = str(Path(p)/"lm_rois")
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(res))

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        if self.worker and self.worker.isRunning():
            ret = QtWidgets.QMessageBox.question(self, APP_TITLE, "Pipeline is running. Quit anyway?")
            if ret != QtWidgets.QMessageBox.Yes:
                e.ignore(); return
        self.save_state()
        return super().closeEvent(e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    # 可选：如果安装了 qfluentwidgets，可以启用 Fluent 深色主题：
    # from qfluentwidgets import FluentIcon as FIF, setTheme, Theme
    # setTheme(Theme.DARK)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

"""
打包提示：
PyInstaller（需在包含依赖的虚拟环境内执行）：
  pyinstaller -F -n KneePipelineStudioQt --clean \
    --add-data "clear_seg_224.py;." --add-data "clear_seg_256.py;." \
    --add-data "merged_clear.py;." --add-data "batch_make_rois.py;." \
    --add-data "split_lm_real1.py;." knee_pipeline_studio_qt.py
若使用 QFluentWidgets，请把其资源按需 --add-data 或确保 pip 安装包在运行环境可见。
许可证注意：
- PySide6 (LGPL-3.0)：动态链接 Qt 库，随二进制保留许可与版权声明，并允许用户替换相同版本的 Qt 库。
- 你自己的业务代码可闭源/商用。
"""
