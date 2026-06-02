#!/usr/bin/env python3
"""
Z-Filter Configuration Tool
============================
Loads a rosbag, reconstructs point clouds from depth + camera_info, and lets
you drag sliders to set min/max Z cutoffs.  A frame scrubber lets you preview
the filter across all frames in the bag.  The live RGB image is shown on the
right so you can see exactly what the robot sees at each frame.

Controls:
  • Z-min / Z-max spinboxes and sliders — adjust the depth crop
  • Frame slider  — scrub through all depth frames in the bag
  • ◀ / ▶ buttons or Left/Right arrow keys — step one frame at a time
  • Save Config — writes z_filter_config.yaml (read by converter + executor)

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
    Index all depth + RGB frame timestamps in the bag without loading pixel data.
    Returns (depth_timestamps, rgb_timestamps, cam_msg, db_path_str,
             depth_topic_id, rgb_topic_id).
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
            all_topics = [r[0] for r in cur.execute("SELECT name FROM topics")]
            raise RuntimeError(f"Topic '{name}' not found in bag.\n"
                               f"Available: {all_topics}")
        return row[0]

    depth_tid   = topic_id(DEPTH_TOPIC)
    caminfo_tid = topic_id(CAMINFO_TOPIC)
    rgb_tid     = topic_id(RGB_TOPIC)

    cur.execute("SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp",
                (depth_tid,))
    depth_ts = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp",
                (rgb_tid,))
    rgb_ts = [r[0] for r in cur.fetchall()]

    if not depth_ts:
        raise RuntimeError("No depth frames found in bag.")

    cur.execute("SELECT data FROM messages WHERE topic_id=? LIMIT 1", (caminfo_tid,))
    cam_row = cur.fetchone()
    if cam_row is None:
        raise RuntimeError("No camera_info found in bag.")
    cam_msg = deserialize_message(bytes(cam_row[0]), CameraInfo)

    conn.close()
    return depth_ts, rgb_ts, cam_msg, db_path, depth_tid, rgb_tid


def _query_nearest_ts(db_path: str, topic_id: int,
                      target_ts: int, msg_type) -> object:
    """Fetch the message with the closest timestamp to target_ts."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    # Try exact match first (fast path)
    cur.execute("SELECT data FROM messages WHERE topic_id=? AND timestamp=?",
                (topic_id, target_ts))
    row = cur.fetchone()
    if row is None:
        # Fall back to nearest neighbour
        cur.execute(
            "SELECT data FROM messages WHERE topic_id=? "
            "ORDER BY ABS(timestamp - ?) LIMIT 1",
            (topic_id, target_ts))
        row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return deserialize_message(bytes(row[0]), msg_type)


def load_depth_at(db_path: str, depth_tid: int, timestamp: int) -> Image:
    return _query_nearest_ts(db_path, depth_tid, timestamp, Image)


def load_rgb_at(db_path: str, rgb_tid: int, timestamp: int) -> Image:
    return _query_nearest_ts(db_path, rgb_tid, timestamp, Image)


# ── image conversion ──────────────────────────────────────────────────────────

def rgb_msg_to_qpixmap(rgb_msg: Image, target_width: int) -> QPixmap:
    """Convert a ROS Image message to a scaled QPixmap."""
    h, w = rgb_msg.height, rgb_msg.width
    raw  = np.frombuffer(bytes(rgb_msg.data), dtype=np.uint8)
    enc  = rgb_msg.encoding

    if enc in ("bgra8", "rgba8"):
        img  = raw.reshape(h, w, 4)
        rgba = img[:, :, [2, 1, 0, 3]].copy() if enc == "bgra8" else img.copy()
        qimg = QImage(rgba.data, w, h, rgba.strides[0], QImage.Format_RGBA8888)
    elif enc == "bgr8":
        img = raw.reshape(h, w, 3)
        rgb = img[:, :, ::-1].copy()   # BGR → RGB
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
    else:  # rgb8
        img  = raw.reshape(h, w, 3).copy()
        qimg = QImage(img.data, w, h, img.strides[0], QImage.Format_RGB888)

    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaledToWidth(target_width, Qt.SmoothTransformation)


# ── point cloud ───────────────────────────────────────────────────────────────

def depth_to_pointcloud(depth_msg: Image, cam_msg: CameraInfo) -> np.ndarray:
    """32FC1 depth image → (N, 3) XYZ point cloud in metres."""
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
    return pts[valid]


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

    RGB_PANEL_W = 340   # fixed width of the right panel (px)

    def __init__(self, depth_ts: list, rgb_ts: list, cam_msg: CameraInfo,
                 db_path: str, depth_tid: int, rgb_tid: int,
                 z_min: float, z_max: float):
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

        self.setWindowTitle("Z-Filter Configuration Tool")
        self.setMinimumSize(1500, 720)
        self._apply_dark()

        # Load first frame data
        self.all_pts = self._fetch_pts(0)
        self._precompute_colors()

        self._build_ui()
        self._update_frame_label()
        self._update_rgb(0)
        self._redraw()

    # ── data loading ──────────────────────────────────────────────────────────

    def _fetch_pts(self, idx: int) -> np.ndarray:
        ts        = self.depth_ts[idx]
        depth_msg = load_depth_at(self.db_path, self.depth_tid, ts)
        if depth_msg is None:
            return np.zeros((0, 3), dtype=np.float32)
        return depth_to_pointcloud(depth_msg, self.cam_msg)

    def _nearest_rgb_ts(self, depth_ts: int) -> int:
        """Return the RGB timestamp closest to a given depth timestamp."""
        if len(self.rgb_ts) == 0:
            return depth_ts
        idx = int(np.argmin(np.abs(self.rgb_ts - depth_ts)))
        return int(self.rgb_ts[idx])

    def _precompute_colors(self):
        pts = self.all_pts
        if len(pts) == 0:
            self.all_colors = np.zeros((0, 3), dtype=np.float32)
            return
        def norm(v):
            mn, mx = v.min(), v.max()
            return (v - mn) / (mx - mn + 1e-6)
        self.all_colors = np.clip(
            np.stack([norm(pts[:, 0]), norm(pts[:, 1]), norm(pts[:, 2])], axis=1),
            0.0, 1.0
        ).astype(np.float32)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        hbox = QHBoxLayout(root)
        hbox.setContentsMargins(10, 10, 10, 10)
        hbox.setSpacing(10)

        # ── left: 3 plots + frame scrubber ───────────────────────────────────
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setSpacing(4)

        self.fig    = Figure(figsize=(12, 4), facecolor="#2d2d2d")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout.addWidget(self.canvas)

        self.ax_top   = self.fig.add_subplot(131)
        self.ax_front = self.fig.add_subplot(132)
        self.ax_hist  = self.fig.add_subplot(133)
        self.fig.tight_layout(pad=2.0)

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
        self.frame_slider.setToolTip("Scrub through frames")
        self.frame_slider.valueChanged.connect(self._on_frame)
        frame_hbox.addWidget(self.frame_slider)

        self.frame_lbl = QLabel()
        self.frame_lbl.setFixedWidth(220)
        self.frame_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.frame_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        frame_hbox.addWidget(self.frame_lbl)

        plot_layout.addWidget(frame_bar)
        hbox.addWidget(plot_widget, stretch=4)

        # ── right panel: RGB image + z controls ──────────────────────────────
        right = QWidget()
        right.setFixedWidth(self.RGB_PANEL_W)
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # RGB image label
        rgb_title = QLabel("RGB Camera")
        rgb_title.setFont(QFont("Arial", 10, QFont.Bold))
        rgb_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(rgb_title)

        self.rgb_lbl = QLabel()
        self.rgb_lbl.setFixedSize(self.RGB_PANEL_W, int(self.RGB_PANEL_W * 9 / 16))
        self.rgb_lbl.setAlignment(Qt.AlignCenter)
        self.rgb_lbl.setStyleSheet("background:#111; border:1px solid #444;")
        right_layout.addWidget(self.rgb_lbl)

        # Thin separator
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
        self.all_pts = self._fetch_pts(idx)
        self._precompute_colors()
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
        """Load the RGB frame nearest to this depth frame and display it."""
        depth_ts  = self.depth_ts[depth_idx]
        nearest   = self._nearest_rgb_ts(depth_ts)
        rgb_msg   = load_rgb_at(self.db_path, self.rgb_tid, nearest)
        if rgb_msg is None:
            self.rgb_lbl.setText("No RGB")
            return
        pixmap = rgb_msg_to_qpixmap(rgb_msg, self.RGB_PANEL_W)
        self.rgb_lbl.setPixmap(pixmap)

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
        z      = pts[:, 2] if len(pts) else np.array([])
        mask   = (z >= self.z_min) & (z <= self.z_max) if len(z) else np.array([], dtype=bool)

        kept    = pts[mask];  kept_col    = colors[mask]
        removed = pts[~mask]; removed_col = colors[~mask]

        for ax in [self.ax_top, self.ax_front, self.ax_hist]:
            ax.cla()
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#555")
            ax.title.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")

        # Top-down: X vs Z
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
        self.ax_top.set_xlabel("X (m)")
        self.ax_top.set_ylabel("Z / depth (m)")
        self.ax_top.set_title("Top-Down  (X vs Depth)")
        self.ax_top.legend(fontsize=7, facecolor="#2d2d2d", labelcolor="white")

        # Front view: X vs Y
        s_rm = max(1, len(removed) // 8000)
        s_kp = max(1, len(kept)    // 8000)
        if len(removed) > 0:
            self.ax_front.scatter(removed[::s_rm, 0], -removed[::s_rm, 1],
                                  s=0.4, c=removed_col[::s_rm] * 0.25, alpha=0.5)
        if len(kept) > 0:
            self.ax_front.scatter(kept[::s_kp, 0], -kept[::s_kp, 1],
                                  s=0.4, c=kept_col[::s_kp], alpha=0.9)
        self.ax_front.set_xlabel("X (m)")
        self.ax_front.set_ylabel("Y (m)  [↑ up]")
        self.ax_front.set_title("Front View  (X vs Y)  RGB = XYZ")

        # Depth histogram
        if len(z) > 0:
            self.ax_hist.hist(z[z < 6.0], bins=120, color="#3498db", alpha=0.7, label="all")
        self.ax_hist.axvline(self.z_min, color="#e67e22", lw=2,
                             label=f"z_min={self.z_min:.2f}")
        self.ax_hist.axvline(self.z_max, color="#e74c3c", lw=2,
                             label=f"z_max={self.z_max:.2f}")
        self.ax_hist.set_xlabel("Depth Z (m)")
        self.ax_hist.set_ylabel("# pixels")
        self.ax_hist.set_title("Depth Histogram")
        self.ax_hist.legend(fontsize=7, facecolor="#2d2d2d", labelcolor="white")

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

        n_kept = int(mask.sum())
        n_all  = len(pts)
        pct    = 100 * n_kept / max(1, n_all)
        self.stats_lbl.setText(
            f"Total pts : {n_all:,}\n"
            f"Kept      : {n_kept:,}  ({pct:.1f}%)\n"
            f"Filtered  : {n_all - n_kept:,}"
        )

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
          f"Duration: {(depth_ts[-1] - depth_ts[0]) / 1e9:.1f}s")
    rclpy.shutdown()

    z_min, z_max = load_config()

    app = QApplication(sys.argv)
    win = ZFilterWindow(depth_ts, rgb_ts, cam_msg, db_path,
                        depth_tid, rgb_tid, z_min, z_max)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
