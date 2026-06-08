#!/usr/bin/env python3
"""
Z-Filter Configuration Tool
============================
Loads a rosbag, reconstructs XYZRGB point clouds from depth + camera_info + RGB,
and lets you drag sliders to set min/max Z cutoffs.

• Point clouds are shown with actual camera RGB colours (not synthetic XYZ mapping).
• A 4th panel shows the 1024-point FPS result (exact DP3 training parameters) —
  click "▶ Run FPS" to trigger it after you've set the z-filter.

Controls:
  • Z-min / Z-max spinboxes and sliders — adjust the depth crop
  • Frame slider / ◀ ▶ / Left–Right arrow keys — scrub through all frames
  • ▶ Run FPS — compute and display the 1024-pt FPS result for the current frame
  • Save Config — writes z_filter_config.yaml (auto-read by converter + executor)

Usage:
    python3 z_filter_gui.py --bag "/media/rosi/T7 Shield/28May/session_20260528_060225_filtered"

Requirements: PyQt5, matplotlib, numpy, rclpy
"""

import argparse
import sqlite3
import sys
import yaml
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QGroupBox, QSizePolicy, QMessageBox,
    QDoubleSpinBox, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor, QImage, QPixmap

CONFIG_PATH   = Path(__file__).parent / "z_filter_config.yaml"
DEPTH_TOPIC   = "/zed/zed_node/depth/depth_registered"
CAMINFO_TOPIC = "/zed/zed_node/depth/camera_info"
RGB_TOPIC     = "/zed/zed_node/rgb/color/rect/image"
DEFAULT_Z_MIN = 0.1
DEFAULT_Z_MAX = 1.5


# ── bag helpers ───────────────────────────────────────────────────────────────

def load_bag_index(bag_dir: Path):
    """
    Index depth + RGB timestamps without loading pixel data.
    Returns (depth_timestamps, rgb_timestamps, cam_msg, db_path, depth_tid, rgb_tid).
    """
    db3 = list(bag_dir.glob("*.db3"))
    if not db3:
        raise RuntimeError(f"No .db3 file found in {bag_dir}")
    db_path = str(db3[0])
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    def topic_id(name):
        cur.execute("SELECT id FROM topics WHERE name=?", (name,))
        row = cur.fetchone()
        if row is None:
            all_t = [r[0] for r in cur.execute("SELECT name FROM topics")]
            raise RuntimeError(f"Topic '{name}' not found.\nAvailable: {all_t}")
        return row[0]

    depth_tid   = topic_id(DEPTH_TOPIC)
    caminfo_tid = topic_id(CAMINFO_TOPIC)
    rgb_tid     = topic_id(RGB_TOPIC)

    cur.execute("SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (depth_tid,))
    depth_ts = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (rgb_tid,))
    rgb_ts = [r[0] for r in cur.fetchall()]

    if not depth_ts:
        raise RuntimeError("No depth frames found in bag.")

    cur.execute("SELECT data FROM messages WHERE topic_id=? LIMIT 1", (caminfo_tid,))
    cam_msg = deserialize_message(bytes(cur.fetchone()[0]), CameraInfo)

    conn.close()
    return depth_ts, rgb_ts, cam_msg, db_path, depth_tid, rgb_tid


def _fetch_raw(db_path: str, topic_id: int, target_ts: int, msg_type):
    """Fetch message closest to target_ts."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT data FROM messages WHERE topic_id=? AND timestamp=?",
                (topic_id, target_ts))
    row = cur.fetchone()
    if row is None:
        cur.execute("SELECT data FROM messages WHERE topic_id=? "
                    "ORDER BY ABS(timestamp - ?) LIMIT 1", (topic_id, target_ts))
        row = cur.fetchone()
    conn.close()
    return deserialize_message(bytes(row[0]), msg_type) if row else None


# ── point cloud ───────────────────────────────────────────────────────────────

def depth_rgb_to_xyzrgb(depth_msg: Image, cam_msg: CameraInfo,
                         rgb_msg: Image) -> np.ndarray:
    """
    Reconstruct (N, 6) XYZRGB point cloud from ZED depth + camera_info + RGB.
    No Z filter applied here — the GUI slider handles that in _redraw.
    RGB is in [0, 1].
    """
    h, w = depth_msg.height, depth_msg.width
    depth = np.frombuffer(bytes(depth_msg.data), dtype=np.float32).reshape(h, w)

    fx = cam_msg.k[0]; fy = cam_msg.k[4]
    cx = cam_msg.k[2]; cy = cam_msg.k[5]

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy

    pts   = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = np.isfinite(pts[:, 2]) & (pts[:, 2] > 0)
    pts   = pts[valid]

    raw = np.frombuffer(bytes(rgb_msg.data), dtype=np.uint8)
    enc = rgb_msg.encoding
    if enc in ("bgra8", "rgba8"):
        img = raw.reshape(h, w, 4)
        r_ch = img[:, :, 2] if enc == "bgra8" else img[:, :, 0]
        g_ch = img[:, :, 1]
        b_ch = img[:, :, 0] if enc == "bgra8" else img[:, :, 2]
    elif enc == "bgr8":
        img  = raw.reshape(h, w, 3)
        r_ch, g_ch, b_ch = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    else:  # rgb8
        img  = raw.reshape(h, w, 3)
        r_ch, g_ch, b_ch = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    r = r_ch.reshape(-1)[valid].astype(np.float32) / 255.0
    g = g_ch.reshape(-1)[valid].astype(np.float32) / 255.0
    b = b_ch.reshape(-1)[valid].astype(np.float32) / 255.0

    return np.column_stack([pts, r, g, b])   # (N, 6)


# ── FPS (exact DP3 parameters) ────────────────────────────────────────────────

FPS_PRESAMPLE = 8192   # matches convert_rosbags_to_zarr.py

def fps_numpy(points: np.ndarray, n_samples: int = 1024) -> np.ndarray:
    """
    Farthest Point Sampling — same implementation used in DP3 training.
    Input:  (N, 6) XYZRGB   Output: (1024, 6) XYZRGB
    """
    n_features = points.shape[1]
    n = len(points)
    if n == 0:
        return np.zeros((n_samples, n_features), dtype=np.float32)
    if n <= n_samples:
        idx = np.random.choice(n, n_samples, replace=True)
        return points[idx].astype(np.float32)

    if n > FPS_PRESAMPLE:
        idx = np.random.choice(n, FPS_PRESAMPLE, replace=False)
        points = points[idx]
        n = FPS_PRESAMPLE

    xyz      = points[:, :3].astype(np.float64)
    selected = np.zeros(n_samples, dtype=np.int64)
    dists    = np.full(n, np.inf, dtype=np.float64)
    selected[0] = np.random.randint(n)
    for i in range(1, n_samples):
        last  = xyz[selected[i - 1]]
        d     = np.sum((xyz - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected[i] = np.argmax(dists)
    return points[selected].astype(np.float32)


# ── image conversion ──────────────────────────────────────────────────────────

def rgb_msg_to_qpixmap(rgb_msg: Image, target_width: int) -> QPixmap:
    h, w = rgb_msg.height, rgb_msg.width
    raw  = np.frombuffer(bytes(rgb_msg.data), dtype=np.uint8)
    enc  = rgb_msg.encoding
    if enc in ("bgra8", "rgba8"):
        img  = raw.reshape(h, w, 4)
        rgba = img[:, :, [2, 1, 0, 3]].copy() if enc == "bgra8" else img.copy()
        qimg = QImage(rgba.data, w, h, rgba.strides[0], QImage.Format_RGBA8888)
    elif enc == "bgr8":
        img  = raw.reshape(h, w, 3)
        rgb  = img[:, :, ::-1].copy()
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
    else:
        img  = raw.reshape(h, w, 3).copy()
        qimg = QImage(img.data, w, h, img.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaledToWidth(target_width, Qt.SmoothTransformation)


# ── config I/O ────────────────────────────────────────────────────────────────

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return float(cfg.get("z_min", DEFAULT_Z_MIN)), float(cfg.get("z_max", DEFAULT_Z_MAX))
    return DEFAULT_Z_MIN, DEFAULT_Z_MAX


def save_config(z_min: float, z_max: float):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump({"z_min": round(z_min, 4), "z_max": round(z_max, 4)}, f)
    print(f"[z_filter] Saved: z_min={z_min:.3f}m  z_max={z_max:.3f}m → {CONFIG_PATH}")


# ── main window ───────────────────────────────────────────────────────────────

class ZFilterWindow(QMainWindow):

    RGB_PANEL_W = 340

    def __init__(self, depth_ts, rgb_ts, cam_msg, db_path,
                 depth_tid, rgb_tid, z_min, z_max):
        super().__init__()

        self.depth_ts      = depth_ts
        self.rgb_ts        = np.array(rgb_ts, dtype=np.int64)
        self.cam_msg       = cam_msg
        self.db_path       = db_path
        self.depth_tid     = depth_tid
        self.rgb_tid       = rgb_tid
        self.z_min         = z_min
        self.z_max         = z_max
        self._t0           = depth_ts[0]
        self.current_frame = 0

        # (N, 6) XYZRGB — all valid depth points, no z-filter applied yet
        self.xyzrgb: np.ndarray = np.zeros((0, 6), dtype=np.float32)
        # (1024, 6) FPS result — populated by Run FPS button
        self.fps_pts: np.ndarray = None

        self.setWindowTitle("Z-Filter Configuration Tool")
        self.setMinimumSize(1600, 720)
        self._apply_dark()

        self._fetch_frame(0)

        self._build_ui()
        self._update_frame_label()
        self._update_rgb(0)
        self._redraw()

    # ── data loading ──────────────────────────────────────────────────────────

    def _nearest_rgb_ts(self, depth_ts: int) -> int:
        if len(self.rgb_ts) == 0:
            return depth_ts
        return int(self.rgb_ts[int(np.argmin(np.abs(self.rgb_ts - depth_ts)))])

    def _fetch_frame(self, idx: int):
        """Load depth + nearest RGB → store as (N, 6) XYZRGB in self.xyzrgb."""
        ts        = self.depth_ts[idx]
        depth_msg = _fetch_raw(self.db_path, self.depth_tid, ts, Image)
        rgb_ts    = self._nearest_rgb_ts(ts)
        rgb_msg   = _fetch_raw(self.db_path, self.rgb_tid, rgb_ts, Image)

        if depth_msg is None or rgb_msg is None:
            self.xyzrgb = np.zeros((0, 6), dtype=np.float32)
        else:
            self.xyzrgb = depth_rgb_to_xyzrgb(depth_msg, self.cam_msg, rgb_msg)

        self.fps_pts = None   # invalidate stale FPS result

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        hbox = QHBoxLayout(root)
        hbox.setContentsMargins(10, 10, 10, 10)
        hbox.setSpacing(10)

        # ── left: 4 plots + frame scrubber ───────────────────────────────────
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setSpacing(4)

        self.fig    = Figure(figsize=(18, 5), facecolor="#e8e8e8",
                             constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout.addWidget(self.canvas)

        self.ax_top   = self.fig.add_subplot(141)   # top-down X vs Z
        self.ax_front = self.fig.add_subplot(142)   # front RGB, X vs Y
        self.ax_hist  = self.fig.add_subplot(143)   # depth histogram
        self.ax_fps   = self.fig.add_subplot(144)   # FPS result

        # Frame scrubber bar
        frame_bar  = QWidget()
        frame_hbox = QHBoxLayout(frame_bar)
        frame_hbox.setContentsMargins(0, 2, 0, 0)
        frame_hbox.setSpacing(6)

        btn_prev = QPushButton("◀")
        btn_prev.setFixedWidth(34)
        btn_prev.setToolTip("Previous frame  (Left arrow)")
        btn_prev.clicked.connect(
            lambda: self._on_frame(max(0, self.current_frame - 1)))
        frame_hbox.addWidget(btn_prev)

        btn_next = QPushButton("▶")
        btn_next.setFixedWidth(34)
        btn_next.setToolTip("Next frame  (Right arrow)")
        btn_next.clicked.connect(
            lambda: self._on_frame(min(len(self.depth_ts) - 1,
                                       self.current_frame + 1)))
        frame_hbox.addWidget(btn_next)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, len(self.depth_ts) - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_frame)
        frame_hbox.addWidget(self.frame_slider)

        self.frame_lbl = QLabel()
        self.frame_lbl.setFixedWidth(220)
        self.frame_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.frame_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        frame_hbox.addWidget(self.frame_lbl)

        plot_layout.addWidget(frame_bar)
        hbox.addWidget(plot_widget, stretch=4)

        # ── right panel ───────────────────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(self.RGB_PANEL_W)
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(0, 0, 0, 0)

        rgb_title = QLabel("RGB Camera")
        rgb_title.setFont(QFont("Arial", 10, QFont.Bold))
        rgb_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(rgb_title)

        self.rgb_lbl = QLabel()
        self.rgb_lbl.setFixedSize(self.RGB_PANEL_W, int(self.RGB_PANEL_W * 9 / 16))
        self.rgb_lbl.setAlignment(Qt.AlignCenter)
        self.rgb_lbl.setStyleSheet("background:#111; border:1px solid #444;")
        right_layout.addWidget(self.rgb_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#555;")
        right_layout.addWidget(sep)

        # z_min
        grp_min = QGroupBox("Min Z (metres)")
        vmin = QVBoxLayout(grp_min)
        self.spin_zmin = QDoubleSpinBox()
        self.spin_zmin.setRange(0.0, 5.0)
        self.spin_zmin.setSingleStep(0.05)
        self.spin_zmin.setDecimals(3)
        self.spin_zmin.setValue(self.z_min)
        self.spin_zmin.valueChanged.connect(self._on_zmin)
        vmin.addWidget(self.spin_zmin)
        self.slider_zmin = QSlider(Qt.Horizontal)
        self.slider_zmin.setRange(0, 500)
        self.slider_zmin.setValue(int(self.z_min * 100))
        self.slider_zmin.valueChanged.connect(
            lambda v: self.spin_zmin.setValue(v / 100))
        vmin.addWidget(self.slider_zmin)
        right_layout.addWidget(grp_min)

        # z_max
        grp_max = QGroupBox("Max Z (metres)")
        vmax = QVBoxLayout(grp_max)
        self.spin_zmax = QDoubleSpinBox()
        self.spin_zmax.setRange(0.0, 10.0)
        self.spin_zmax.setSingleStep(0.05)
        self.spin_zmax.setDecimals(3)
        self.spin_zmax.setValue(self.z_max)
        self.spin_zmax.valueChanged.connect(self._on_zmax)
        vmax.addWidget(self.spin_zmax)
        self.slider_zmax = QSlider(Qt.Horizontal)
        self.slider_zmax.setRange(0, 1000)
        self.slider_zmax.setValue(int(self.z_max * 100))
        self.slider_zmax.valueChanged.connect(
            lambda v: self.spin_zmax.setValue(v / 100))
        vmax.addWidget(self.slider_zmax)
        right_layout.addWidget(grp_max)

        # Stats
        self.stats_lbl = QLabel("")
        self.stats_lbl.setAlignment(Qt.AlignCenter)
        self.stats_lbl.setWordWrap(True)
        self.stats_lbl.setStyleSheet("color:#1abc9c; font-size:11px;")
        right_layout.addWidget(self.stats_lbl)

        # FPS preview group
        grp_fps = QGroupBox("FPS Preview")
        fps_layout = QVBoxLayout(grp_fps)
        fps_layout.setSpacing(5)

        # Quick-select preset buttons
        presets_row = QHBoxLayout()
        presets_row.setSpacing(3)
        for n in (512, 1024, 2048, 4096, 8192):
            btn = QPushButton(str(n))
            btn.setFixedHeight(24)
            btn.setStyleSheet(
                "QPushButton{background:#3a3a3a;color:#ddd;border-radius:3px;"
                "font-size:10px;border:1px solid #555;}"
                "QPushButton:hover{background:#555;}"
            )
            btn.clicked.connect(lambda checked, v=n: self._set_fps_n(v))
            presets_row.addWidget(btn)
        fps_layout.addLayout(presets_row)

        # Custom spinbox
        custom_row = QHBoxLayout()
        custom_row.setSpacing(5)
        custom_lbl = QLabel("Custom:")
        custom_lbl.setStyleSheet("color:#ccc; font-size:10px;")
        custom_lbl.setFixedWidth(52)
        custom_row.addWidget(custom_lbl)
        from PyQt5.QtWidgets import QSpinBox
        self.fps_n_spin = QSpinBox()
        self.fps_n_spin.setRange(64, 65536)
        self.fps_n_spin.setSingleStep(512)
        self.fps_n_spin.setValue(1024)
        self.fps_n_spin.setStyleSheet("color:white; background:#2a2a2a;")
        custom_row.addWidget(self.fps_n_spin)
        fps_layout.addLayout(custom_row)

        # Run button
        self.fps_btn = QPushButton("▶  Run FPS")
        self.fps_btn.setMinimumHeight(34)
        self.fps_btn.setStyleSheet(
            "QPushButton{background:#2980b9;color:white;border-radius:5px;"
            "font-size:11px;font-weight:bold;}"
            "QPushButton:hover{background:#3498db;}"
            "QPushButton:disabled{background:#555;color:#888;}"
        )
        self.fps_btn.clicked.connect(self._run_fps)
        fps_layout.addWidget(self.fps_btn)

        right_layout.addWidget(grp_fps)

        right_layout.addStretch()

        # Save button
        self.save_btn = QPushButton("💾  Save Config")
        self.save_btn.setMinimumHeight(44)
        self.save_btn.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;border-radius:5px;"
            "font-size:12px;font-weight:bold;}"
            "QPushButton:hover{background:#2ecc71;}"
        )
        self.save_btn.clicked.connect(self._save)
        right_layout.addWidget(self.save_btn)

        hbox.addWidget(right)

    def _apply_dark(self):
        app = QApplication.instance()
        app.setStyle("Fusion")
        pal = QPalette()
        pal.setColor(QPalette.Window,     QColor(45, 45, 45))
        pal.setColor(QPalette.WindowText, Qt.white)
        pal.setColor(QPalette.Base,       QColor(30, 30, 30))
        pal.setColor(QPalette.Text,       Qt.white)
        pal.setColor(QPalette.Button,     QColor(55, 55, 55))
        pal.setColor(QPalette.ButtonText, Qt.white)
        app.setPalette(pal)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self._on_frame(max(0, self.current_frame - 1))
        elif event.key() == Qt.Key_Right:
            self._on_frame(min(len(self.depth_ts) - 1, self.current_frame + 1))
        else:
            super().keyPressEvent(event)

    def _on_frame(self, idx: int):
        if idx == self.current_frame:
            return
        self.current_frame = idx
        self._fetch_frame(idx)
        self._update_frame_label()
        self._update_rgb(idx)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(idx)
        self.frame_slider.blockSignals(False)
        self._redraw()

    def _update_frame_label(self):
        t_sec = (self.depth_ts[self.current_frame] - self._t0) / 1e9
        self.frame_lbl.setText(
            f"Frame {self.current_frame + 1} / {len(self.depth_ts)}  "
            f"(t={t_sec:.1f}s)")

    def _update_rgb(self, depth_idx: int):
        rgb_ts  = self._nearest_rgb_ts(self.depth_ts[depth_idx])
        rgb_msg = _fetch_raw(self.db_path, self.rgb_tid, rgb_ts, Image)
        if rgb_msg is None:
            self.rgb_lbl.setText("No RGB")
            return
        self.rgb_lbl.setPixmap(rgb_msg_to_qpixmap(rgb_msg, self.RGB_PANEL_W))

    def _on_zmin(self, v):
        self.z_min = v
        self.slider_zmin.blockSignals(True)
        self.slider_zmin.setValue(int(v * 100))
        self.slider_zmin.blockSignals(False)
        self.fps_pts = None   # z changed → FPS result stale
        self._redraw()

    def _on_zmax(self, v):
        self.z_max = v
        self.slider_zmax.blockSignals(True)
        self.slider_zmax.setValue(int(v * 100))
        self.slider_zmax.blockSignals(False)
        self.fps_pts = None
        self._redraw()

    def _set_fps_n(self, n: int):
        """Called by preset buttons — update the spinbox and re-run if a result exists."""
        self.fps_n_spin.setValue(n)
        if self.fps_pts is not None:
            self._run_fps()

    def _run_fps(self):
        """Apply z-filter to the current cloud then run FPS with the selected n_pts."""
        pts  = self.xyzrgb
        if len(pts) == 0:
            return
        z    = pts[:, 2]
        kept = pts[(z >= self.z_min) & (z <= self.z_max)]
        if len(kept) == 0:
            return

        n_pts = self.fps_n_spin.value()
        self.fps_btn.setEnabled(False)
        self.fps_btn.setText(f"Computing {n_pts:,} pts…")
        QApplication.processEvents()

        self.fps_pts = fps_numpy(kept, n_pts)

        self.fps_btn.setEnabled(True)
        self.fps_btn.setText("▶  Run FPS")
        self._draw_fps_panel()
        self.canvas.draw()

    # ── drawing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _style_ax(ax, title: str, xlabel: str, ylabel: str):
        ax.cla()
        ax.set_facecolor("white")
        ax.tick_params(colors="#333333")
        for spine in ax.spines.values():
            spine.set_edgecolor("#bbbbbb")
        ax.set_title(title,   color="#111111", fontsize=9, fontweight="bold")
        ax.set_xlabel(xlabel, color="#333333", fontsize=8)
        ax.set_ylabel(ylabel, color="#333333", fontsize=8)

    def _redraw(self):
        pts = self.xyzrgb
        if len(pts) == 0:
            for ax in [self.ax_top, self.ax_front, self.ax_hist, self.ax_fps]:
                self._style_ax(ax, "No data", "", "")
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="#aaaaaa", fontsize=10)
            self.canvas.draw()
            return

        z    = pts[:, 2]
        mask = (z >= self.z_min) & (z <= self.z_max)
        kept    = pts[mask]
        removed = pts[~mask]

        # subsample for scatter performance
        def sub(arr, cap=6000):
            s = max(1, len(arr) // cap)
            return arr[::s]

        # ── 1. Top-down: X vs Z, actual RGB for kept, dim red for removed ────
        self._style_ax(self.ax_top, "Top-Down  (X vs Depth)",
                       "X (m)", "Z / depth (m)")
        if len(removed) > 0:
            r = sub(removed)
            self.ax_top.scatter(r[:, 0], r[:, 2],
                                s=0.3, c="red", alpha=0.35, label="filtered")
        if len(kept) > 0:
            k = sub(kept)
            self.ax_top.scatter(k[:, 0], k[:, 2],
                                s=0.3, c=k[:, 3:6].clip(0, 1), alpha=0.7, label="kept")
        self.ax_top.axhline(self.z_min, color="#d35400", lw=1.5, ls="--",
                            label=f"z_min={self.z_min:.2f}")
        self.ax_top.axhline(self.z_max, color="#c0392b", lw=1.5, ls="--",
                            label=f"z_max={self.z_max:.2f}")
        self.ax_top.legend(fontsize=6, facecolor="white", labelcolor="#222222",
                           edgecolor="#cccccc")
        self.ax_top.set_aspect('equal', adjustable='datalim')

        # ── 2. Front view: X vs Y, actual RGB ────────────────────────────────
        self._style_ax(self.ax_front, "Front View  (actual RGB)",
                       "X (m)", "Y (m)  [↑ up]")
        if len(removed) > 0:
            r = sub(removed, 8000)
            gray = np.full((len(r), 3), 0.25, dtype=np.float32)
            self.ax_front.scatter(r[:, 0], -r[:, 1],
                                  s=0.4, c=gray, alpha=0.4)
        if len(kept) > 0:
            k = sub(kept, 8000)
            self.ax_front.scatter(k[:, 0], -k[:, 1],
                                  s=0.5, c=k[:, 3:6].clip(0, 1), alpha=0.95)
        self.ax_front.set_aspect('equal', adjustable='datalim')

        # ── 3. Depth histogram ────────────────────────────────────────────────
        self._style_ax(self.ax_hist, "Depth Histogram",
                       "Z (m)", "# pixels")
        z_clip = z[z < 6.0]
        if len(z_clip) > 0:
            self.ax_hist.hist(z_clip, bins=120, color="#4a90d9", alpha=0.75,
                              edgecolor="white", linewidth=0.3)
        self.ax_hist.axvline(self.z_min, color="#d35400", lw=2,
                             label=f"z_min={self.z_min:.2f}")
        self.ax_hist.axvline(self.z_max, color="#c0392b", lw=2,
                             label=f"z_max={self.z_max:.2f}")
        self.ax_hist.legend(fontsize=7, facecolor="white", labelcolor="#222222",
                            edgecolor="#cccccc")

        # ── 4. FPS panel ──────────────────────────────────────────────────────
        self._draw_fps_panel()

        self.canvas.draw()

        n_kept = int(mask.sum())
        n_all  = len(pts)
        self.stats_lbl.setText(
            f"Total pts : {n_all:,}\n"
            f"Kept      : {n_kept:,}  ({100*n_kept/max(1,n_all):.1f}%)\n"
            f"Filtered  : {n_all - n_kept:,}"
        )

    def _draw_fps_panel(self):
        n_sel = self.fps_n_spin.value() if hasattr(self, 'fps_n_spin') else 1024
        if self.fps_pts is None:
            self._style_ax(self.ax_fps,
                           f"FPS Preview  (select n then ▶ Run FPS)",
                           "X (m)", "Y (m)  [↑ up]")
            self.ax_fps.text(
                0.5, 0.5,
                "Set z-filter, choose n pts,\nthen click  ▶ Run FPS",
                transform=self.ax_fps.transAxes,
                ha="center", va="center",
                color="#888888", fontsize=9
            )
        else:
            n_actual = len(self.fps_pts)
            dp3_note = "  ← DP3 default" if n_actual == 1024 else ""
            self._style_ax(self.ax_fps,
                           f"FPS Result  ({n_actual:,} pts{dp3_note})",
                           "X (m)", "Y (m)  [↑ up]")
            p = self.fps_pts
            pt_size = max(1.0, min(6.0, 4096 / max(n_actual, 1)))
            self.ax_fps.scatter(p[:, 0], -p[:, 1],
                                s=pt_size, c=p[:, 3:6].clip(0, 1), alpha=0.95)
            self.ax_fps.set_aspect('equal', adjustable='datalim')

    # ── save ──────────────────────────────────────────────────────────────────

    def _save(self):
        if self.z_min >= self.z_max:
            QMessageBox.warning(self, "Invalid Range", "z_min must be less than z_max.")
            return
        save_config(self.z_min, self.z_max)
        QMessageBox.information(
            self, "Saved",
            f"Config saved to:\n{CONFIG_PATH}\n\n"
            f"z_min = {self.z_min:.3f} m\n"
            f"z_max = {self.z_max:.3f} m"
        )


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag",
        default="/media/rosi/T7 Shield/28May/session_20260528_060225_filtered",
        help="Path to a rosbag folder containing ZED depth + RGB + camera_info topics",
    )
    args = parser.parse_args()

    rclpy.init(args=None)
    bag_dir = Path(args.bag)
    print(f"Indexing bag: {bag_dir} ...")
    depth_ts, rgb_ts, cam_msg, db_path, depth_tid, rgb_tid = load_bag_index(bag_dir)
    print(f"Found {len(depth_ts)} depth frames, {len(rgb_ts)} RGB frames.  "
          f"Duration: {(depth_ts[-1]-depth_ts[0])/1e9:.1f}s")
    rclpy.shutdown()

    z_min, z_max = load_config()

    app = QApplication(sys.argv)
    win = ZFilterWindow(depth_ts, rgb_ts, cam_msg, db_path,
                        depth_tid, rgb_tid, z_min, z_max)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
