#!/usr/bin/env python3
"""
Z-Filter Configuration Tool
============================
Loads one rosbag, reconstructs the point cloud from depth + camera_info,
and lets you drag sliders to set min/max Z cutoffs.  Saves the result to
z_filter_config.yaml which is read by both rosbag_to_zarr.py and
policy_executor.py at runtime.

Usage:
    python3 z_filter_gui.py --bag /home/rosi/rosbags_96221/session_20260429_164113

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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QGroupBox, QSizePolicy, QMessageBox,
    QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor

CONFIG_PATH = Path(__file__).parent / "z_filter_config.yaml"
DEFAULT_Z_MIN = 0.1   # metres
DEFAULT_Z_MAX = 1.5   # metres


# ── helpers ──────────────────────────────────────────────────────────────────

def load_frame(bag_dir: Path):
    """Load one depth frame + camera info from a bag."""
    db3 = list(bag_dir.glob("*.db3"))[0]
    conn = sqlite3.connect(str(db3))
    cur = conn.cursor()

    def get_msg(topic, msg_type):
        cur.execute("SELECT id FROM topics WHERE name=?", (topic,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"Topic {topic} not in bag")
        tid = row[0]
        # grab mid-episode frame for a representative view
        cur.execute("SELECT data FROM messages WHERE topic_id=? LIMIT 1 OFFSET 5", (tid,))
        raw = cur.fetchone()
        if raw is None:
            cur.execute("SELECT data FROM messages WHERE topic_id=? LIMIT 1", (tid,))
            raw = cur.fetchone()
        return deserialize_message(bytes(raw[0]), msg_type)

    depth_msg  = get_msg("/zed/zed_node/depth/depth_registered", Image)
    cam_msg    = get_msg("/zed/zed_node/depth/camera_info",      CameraInfo)
    conn.close()
    return depth_msg, cam_msg


def depth_to_pointcloud(depth_msg: Image, cam_msg: CameraInfo):
    """Convert a 32FC1 depth image to an Nx3 point cloud (metres)."""
    h, w = depth_msg.height, depth_msg.width
    depth = np.frombuffer(bytes(depth_msg.data), dtype=np.float32).reshape(h, w)

    fx = cam_msg.k[0]; fy = cam_msg.k[4]
    cx = cam_msg.k[2]; cy = cam_msg.k[5]

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)

    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy

    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = np.isfinite(pts[:, 2]) & (pts[:, 2] > 0)
    return pts[valid]


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return float(cfg.get("z_min", DEFAULT_Z_MIN)), float(cfg.get("z_max", DEFAULT_Z_MAX))
    return DEFAULT_Z_MIN, DEFAULT_Z_MAX


def save_config(z_min: float, z_max: float):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump({"z_min": round(z_min, 4), "z_max": round(z_max, 4)}, f)
    print(f"[z_filter] Saved config: z_min={z_min:.3f}m  z_max={z_max:.3f}m → {CONFIG_PATH}")


# ── main window ───────────────────────────────────────────────────────────────

class ZFilterWindow(QMainWindow):

    def __init__(self, all_pts: np.ndarray, z_min: float, z_max: float):
        super().__init__()
        self.all_pts = all_pts
        self.z_min   = z_min
        self.z_max   = z_max
        self._precompute_colors()

        self.setWindowTitle("Z-Filter Configuration Tool")
        self.setMinimumSize(1350, 650)
        self._apply_dark()
        self._build_ui()
        self._redraw()

    def _precompute_colors(self):
        """Compute per-point RGB from XYZ (normalized over full cloud, stable while sliding)."""
        pts = self.all_pts
        def norm(v):
            mn, mx = v.min(), v.max()
            return (v - mn) / (mx - mn + 1e-6)
        self.all_colors = np.clip(
            np.stack([norm(pts[:, 0]), norm(pts[:, 1]), norm(pts[:, 2])], axis=1),
            0.0, 1.0
        ).astype(np.float32)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        hbox = QHBoxLayout(root)
        hbox.setContentsMargins(10, 10, 10, 10)

        # ── left: plots ──
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        self.fig = Figure(figsize=(12, 4), facecolor="#2d2d2d")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout.addWidget(self.canvas)

        self.ax_top   = self.fig.add_subplot(131)   # top-down  (X vs Z)
        self.ax_front = self.fig.add_subplot(132)   # front RGB (X vs Y)
        self.ax_hist  = self.fig.add_subplot(133)   # depth histogram
        self.fig.tight_layout(pad=2.0)

        hbox.addWidget(plot_widget, stretch=4)

        # ── right: controls ──
        ctrl = QWidget()
        ctrl.setFixedWidth(230)
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setSpacing(12)

        title = QLabel("Z-Filter Setup")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        ctrl_layout.addWidget(title)

        info = QLabel(
            "Adjust min/max Z to remove\n"
            "background and floor.\n\n"
            "Top-Down:  Blue=kept  Red=filtered\n"
            "Front RGB: X→R  Y→G  Z→B\n"
            "  (bright = kept, dim = filtered)\n\n"
            "Histogram: look for a gap\n"
            "after the workspace depth."
        )
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignCenter)
        ctrl_layout.addWidget(info)

        # -- z_min spinner
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
        self.slider_zmin.valueChanged.connect(lambda v: self.spin_zmin.setValue(v / 100))
        vmin.addWidget(self.slider_zmin)
        ctrl_layout.addWidget(grp_min)

        # -- z_max spinner
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
        self.slider_zmax.valueChanged.connect(lambda v: self.spin_zmax.setValue(v / 100))
        vmax.addWidget(self.slider_zmax)
        ctrl_layout.addWidget(grp_max)

        # -- stats label
        self.stats_lbl = QLabel("")
        self.stats_lbl.setAlignment(Qt.AlignCenter)
        self.stats_lbl.setWordWrap(True)
        self.stats_lbl.setStyleSheet("color:#1abc9c; font-size:11px;")
        ctrl_layout.addWidget(self.stats_lbl)

        ctrl_layout.addStretch()

        # -- save button
        self.save_btn = QPushButton("💾  Save Config")
        self.save_btn.setMinimumHeight(44)
        self.save_btn.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;border-radius:5px;"
            "font-size:12px;font-weight:bold;}"
            "QPushButton:hover{background:#2ecc71;}"
        )
        self.save_btn.clicked.connect(self._save)
        ctrl_layout.addWidget(self.save_btn)

        hbox.addWidget(ctrl, stretch=1)

    def _apply_dark(self):
        app = QApplication.instance()
        app.setStyle("Fusion")
        pal = QPalette()
        pal.setColor(QPalette.Window,        QColor(45, 45, 45))
        pal.setColor(QPalette.WindowText,    Qt.white)
        pal.setColor(QPalette.Base,          QColor(30, 30, 30))
        pal.setColor(QPalette.Text,          Qt.white)
        pal.setColor(QPalette.Button,        QColor(55, 55, 55))
        pal.setColor(QPalette.ButtonText,    Qt.white)
        app.setPalette(pal)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_zmin(self, v):
        self.z_min = v
        self.slider_zmin.blockSignals(True)
        self.slider_zmin.setValue(int(v * 100))
        self.slider_zmin.blockSignals(False)
        self._redraw()

    def _on_zmax(self, v):
        self.z_max = v
        self.slider_zmax.blockSignals(True)
        self.slider_zmax.setValue(int(v * 100))
        self.slider_zmax.blockSignals(False)
        self._redraw()

    def _redraw(self):
        pts    = self.all_pts
        colors = self.all_colors
        z      = pts[:, 2]
        mask   = (z >= self.z_min) & (z <= self.z_max)

        kept    = pts[mask];    kept_col    = colors[mask]
        removed = pts[~mask];   removed_col = colors[~mask]

        for ax in [self.ax_top, self.ax_front, self.ax_hist]:
            ax.cla()
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#555")
            ax.title.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")

        # ── Top-down view: X vs Z ─────────────────────────────────────────────
        s_rm = max(1, len(removed) // 5000)
        s_kp = max(1, len(kept)    // 5000)
        if len(removed) > 0:
            self.ax_top.scatter(removed[::s_rm, 0], removed[::s_rm, 2],
                                s=0.3, c="red",  alpha=0.4, label="filtered")
        if len(kept) > 0:
            self.ax_top.scatter(kept[::s_kp, 0], kept[::s_kp, 2],
                                s=0.3, c="cyan", alpha=0.6, label="kept")
        self.ax_top.axhline(self.z_min, color="#e67e22", lw=1.5, ls="--",
                            label=f"z_min={self.z_min:.2f}")
        self.ax_top.axhline(self.z_max, color="#e74c3c", lw=1.5, ls="--",
                            label=f"z_max={self.z_max:.2f}")
        self.ax_top.set_xlabel("X (m)"); self.ax_top.set_ylabel("Z / depth (m)")
        self.ax_top.set_title("Top-Down  (X vs Depth)")
        self.ax_top.legend(fontsize=7, facecolor="#2d2d2d", labelcolor="white")

        # ── Front view: X vs Y, RGB-XYZ coloring ─────────────────────────────
        # Camera Y is positive-down, so we negate Y so up=up in the plot.
        s_rm = max(1, len(removed) // 8000)
        s_kp = max(1, len(kept)    // 8000)
        if len(removed) > 0:
            dim_col = removed_col[::s_rm] * 0.25   # dim out filtered points
            self.ax_front.scatter(removed[::s_rm, 0], -removed[::s_rm, 1],
                                  s=0.4, c=dim_col, alpha=0.5)
        if len(kept) > 0:
            self.ax_front.scatter(kept[::s_kp, 0], -kept[::s_kp, 1],
                                  s=0.4, c=kept_col[::s_kp], alpha=0.9)
        self.ax_front.set_xlabel("X (m)"); self.ax_front.set_ylabel("Y (m)  [↑ up]")
        self.ax_front.set_title("Front View  (X vs Y)  RGB = XYZ")

        # ── Depth histogram ───────────────────────────────────────────────────
        z_clip = z[z < 6.0]
        self.ax_hist.hist(z_clip, bins=120, color="#3498db", alpha=0.7, label="all")
        self.ax_hist.axvline(self.z_min, color="#e67e22", lw=2,
                             label=f"z_min={self.z_min:.2f}")
        self.ax_hist.axvline(self.z_max, color="#e74c3c", lw=2,
                             label=f"z_max={self.z_max:.2f}")
        self.ax_hist.set_xlabel("Depth Z (m)"); self.ax_hist.set_ylabel("# pixels")
        self.ax_hist.set_title("Depth Histogram")
        self.ax_hist.legend(fontsize=7, facecolor="#2d2d2d", labelcolor="white")

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

        pct = 100 * mask.sum() / max(1, len(pts))
        self.stats_lbl.setText(
            f"Total pts : {len(pts):,}\n"
            f"Kept      : {mask.sum():,}  ({pct:.1f}%)\n"
            f"Filtered  : {(~mask).sum():,}"
        )

    def _save(self):
        if self.z_min >= self.z_max:
            QMessageBox.warning(self, "Invalid Range", "z_min must be less than z_max.")
            return
        save_config(self.z_min, self.z_max)
        QMessageBox.information(self, "Saved",
                                f"Config saved to:\n{CONFIG_PATH}\n\n"
                                f"z_min = {self.z_min:.3f} m\n"
                                f"z_max = {self.z_max:.3f} m")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", default="/home/rosi/DP3_ROSBAGS/session_20260508_081513_filtered",
                        help="Path to one rosbag folder")
    args = parser.parse_args()

    rclpy.init(args=None)
    print(f"Loading depth frame from {args.bag} ...")
    depth_msg, cam_msg = load_frame(Path(args.bag))
    all_pts = depth_to_pointcloud(depth_msg, cam_msg)
    print(f"Point cloud has {len(all_pts):,} points before filtering.")
    rclpy.shutdown()

    z_min, z_max = load_config()

    app = QApplication(sys.argv)
    win = ZFilterWindow(all_pts, z_min, z_max)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
