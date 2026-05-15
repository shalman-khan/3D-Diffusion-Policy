#!/usr/bin/env python3
"""
RGB Point Cloud Viewer + Z-Filter Configuration Tool
======================================================
Loads one rosbag, reconstructs a colored (RGB) point cloud from
depth + rgb + camera_info, and lets you tune Z-filter sliders
interactively in Open3D's native GUI.

Controls:
  Left-drag   : rotate
  Right-drag  : pan
  Scroll      : zoom
  Sliders     : adjust z_min / z_max live
  Save button : write z_filter_config.yaml

Usage:
    python3 rgb_pointcloud_gui.py \
        --bag /home/rosi/rosbags_96221/session_20260429_164113 \
        [--frame 10]   # which frame index to preview (default: mid-episode)
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import yaml
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo

CONFIG_PATH = Path(__file__).parent / "z_filter_config.yaml"
DEFAULT_Z_MIN = 0.1
DEFAULT_Z_MAX = 1.5


# ── data loading ──────────────────────────────────────────────────────────────

def load_frame(bag_dir: Path, frame_idx: int):
    db3  = list(bag_dir.glob("*.db3"))[0]
    conn = sqlite3.connect(str(db3))
    cur  = conn.cursor()

    def get_msg(topic, msg_type, idx=0):
        cur.execute("SELECT id FROM topics WHERE name=?", (topic,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"Topic not found: {topic}")
        tid = row[0]
        cur.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1 OFFSET ?",
            (tid, idx))
        raw = cur.fetchone()
        if raw is None:          # idx beyond end — take last
            cur.execute("SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp DESC LIMIT 1", (tid,))
            raw = cur.fetchone()
        return deserialize_message(bytes(raw[0]), msg_type)

    depth_msg = get_msg("/zed/zed_node/depth/depth_registered", Image,      frame_idx)
    rgb_msg   = get_msg("/zed/zed_node/rgb/color/rect/image",   Image,      frame_idx)
    cam_msg   = get_msg("/zed/zed_node/depth/camera_info",      CameraInfo, 0)
    conn.close()
    return depth_msg, rgb_msg, cam_msg


def build_rgb_pointcloud(depth_msg: Image, rgb_msg: Image, cam_msg: CameraInfo,
                          z_min: float, z_max: float) -> o3d.geometry.PointCloud:
    h, w = depth_msg.height, depth_msg.width

    # Depth (32FC1 metres)
    depth = np.frombuffer(bytes(depth_msg.data), dtype=np.float32).reshape(h, w)

    # RGB from BGRA8
    bgra = np.frombuffer(bytes(rgb_msg.data), dtype=np.uint8).reshape(h, w, 4)
    rgb  = bgra[:, :, [2, 1, 0]].astype(np.float32) / 255.0   # BGR→RGB, normalise

    # Camera intrinsics
    fx = cam_msg.k[0]; fy = cam_msg.k[4]
    cx = cam_msg.k[2]; cy = cam_msg.k[5]

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy

    pts    = np.stack([x, y, z],    axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)

    valid  = (np.isfinite(pts[:, 2])
              & (pts[:, 2] >= z_min)
              & (pts[:, 2] <= z_max))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[valid].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors[valid].astype(np.float64))
    return pcd


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return float(cfg.get("z_min", DEFAULT_Z_MIN)), float(cfg.get("z_max", DEFAULT_Z_MAX))
    return DEFAULT_Z_MIN, DEFAULT_Z_MAX


def save_config(z_min: float, z_max: float):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump({"z_min": round(z_min, 4), "z_max": round(z_max, 4)}, f)
    print(f"[z_filter] Saved → {CONFIG_PATH}  (z_min={z_min:.3f}m, z_max={z_max:.3f}m)")


# ── Open3D GUI app ────────────────────────────────────────────────────────────

class RGBPointCloudApp:

    MATERIAL_NAME = "defaultUnlit"

    def __init__(self, depth_msg, rgb_msg, cam_msg, z_min, z_max):
        self.depth_msg = depth_msg
        self.rgb_msg   = rgb_msg
        self.cam_msg   = cam_msg
        self.z_min     = z_min
        self.z_max     = z_max

        app = gui.Application.instance
        app.initialize()

        self.win = app.create_window("RGB Point Cloud — Z-Filter Setup", 1280, 800)
        self.win.set_on_layout(self._on_layout)
        self.win.set_on_close(self._on_close)

        em = self.win.theme.font_size

        # ── 3D scene ──
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.win.renderer)
        self.scene_widget.scene.set_background([0.15, 0.15, 0.15, 1.0])
        self.scene_widget.scene.show_axes(True)
        self.win.add_child(self.scene_widget)

        # ── right panel ──
        self.panel = gui.Vert(0, gui.Margins(em, em, em, em))
        self.panel.preferred_width = 280

        # Title
        title = gui.Label("Z-Filter Configuration")
        self.panel.add_child(title)
        self.panel.add_fixed(em)

        # Stats label
        self.stats_label = gui.Label("Points: --")
        self.panel.add_child(self.stats_label)
        self.panel.add_fixed(em)

        self.panel.add_child(gui.Label("─" * 30))
        self.panel.add_fixed(em // 2)

        # z_min slider
        self.panel.add_child(gui.Label("Min Z (metres)"))
        self.zmin_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.zmin_edit.set_limits(0.0, 5.0)
        self.zmin_edit.decimal_precision = 3
        self.zmin_edit.double_value = self.z_min
        self.zmin_edit.set_on_value_changed(self._on_zmin)
        self.panel.add_child(self.zmin_edit)

        self.zmin_slider = gui.Slider(gui.Slider.DOUBLE)
        self.zmin_slider.set_limits(0.0, 3.0)
        self.zmin_slider.double_value = self.z_min
        self.zmin_slider.set_on_value_changed(self._on_zmin_slider)
        self.panel.add_child(self.zmin_slider)
        self.panel.add_fixed(em)

        # z_max slider
        self.panel.add_child(gui.Label("Max Z (metres)"))
        self.zmax_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.zmax_edit.set_limits(0.0, 10.0)
        self.zmax_edit.decimal_precision = 3
        self.zmax_edit.double_value = self.z_max
        self.zmax_edit.set_on_value_changed(self._on_zmax)
        self.panel.add_child(self.zmax_edit)

        self.zmax_slider = gui.Slider(gui.Slider.DOUBLE)
        self.zmax_slider.set_limits(0.0, 5.0)
        self.zmax_slider.double_value = self.z_max
        self.zmax_slider.set_on_value_changed(self._on_zmax_slider)
        self.panel.add_child(self.zmax_slider)
        self.panel.add_fixed(em)

        self.panel.add_child(gui.Label("─" * 30))
        self.panel.add_fixed(em // 2)

        # Point size
        self.panel.add_child(gui.Label("Point Size"))
        self.pt_size_slider = gui.Slider(gui.Slider.INT)
        self.pt_size_slider.set_limits(1, 6)
        self.pt_size_slider.int_value = 2
        self.pt_size_slider.set_on_value_changed(self._on_pt_size)
        self.panel.add_child(self.pt_size_slider)
        self.panel.add_fixed(em)

        # Reset view button
        reset_btn = gui.Button("Reset View")
        reset_btn.set_on_clicked(self._reset_view)
        self.panel.add_child(reset_btn)
        self.panel.add_fixed(em // 2)

        # Save button
        save_btn = gui.Button("💾  Save Config")
        save_btn.background_color = gui.Color(0.15, 0.6, 0.25)
        save_btn.set_on_clicked(self._save)
        self.panel.add_child(save_btn)
        self.panel.add_fixed(em // 2)

        self.status_label = gui.Label("")
        self.panel.add_child(self.status_label)

        self.win.add_child(self.panel)

        # Initial render
        self._rebuild_cloud()

    # ── layout ────────────────────────────────────────────────────────────────

    def _on_layout(self, layout_context):
        r = self.win.content_rect
        panel_w = self.panel.preferred_width
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)
        self.panel.frame = gui.Rect(r.x + r.width - panel_w, r.y, panel_w, r.height)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_zmin(self, value):
        self.z_min = value
        self.zmin_slider.double_value = value
        self._rebuild_cloud()

    def _on_zmin_slider(self, value):
        self.z_min = value
        self.zmin_edit.double_value = value
        self._rebuild_cloud()

    def _on_zmax(self, value):
        self.z_max = value
        self.zmax_slider.double_value = value
        self._rebuild_cloud()

    def _on_zmax_slider(self, value):
        self.z_max = value
        self.zmax_edit.double_value = value
        self._rebuild_cloud()

    def _on_pt_size(self, value):
        mat = rendering.MaterialRecord()
        mat.shader = self.MATERIAL_NAME
        mat.point_size = float(int(value))
        self.scene_widget.scene.modify_geometry_material("cloud", mat)
        self.win.post_redraw()

    def _reset_view(self):
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())
        self.win.post_redraw()

    def _save(self):
        if self.z_min >= self.z_max:
            dlg = gui.Dialog("Error")
            dlg.add_child(gui.Label("z_min must be less than z_max."))
            ok = gui.Button("OK")
            ok.set_on_clicked(self.win.close_dialog)
            dlg.add_child(ok)
            self.win.show_dialog(dlg)
            return
        save_config(self.z_min, self.z_max)
        self.status_label.text = f"Saved  z_min={self.z_min:.3f}  z_max={self.z_max:.3f}"

    def _on_close(self):
        gui.Application.instance.quit()
        return True

    # ── cloud rebuild ─────────────────────────────────────────────────────────

    def _rebuild_cloud(self):
        pcd = build_rgb_pointcloud(
            self.depth_msg, self.rgb_msg, self.cam_msg,
            self.z_min, self.z_max)

        n = len(pcd.points)
        self.stats_label.text = (
            f"Points : {n:,}\n"
            f"z_min  : {self.z_min:.3f} m\n"
            f"z_max  : {self.z_max:.3f} m"
        )

        mat = rendering.MaterialRecord()
        mat.shader = self.MATERIAL_NAME
        mat.point_size = float(self.pt_size_slider.int_value)

        self.scene_widget.scene.clear_geometry()
        if n > 0:
            self.scene_widget.scene.add_geometry("cloud", pcd, mat)
            bounds = self.scene_widget.scene.bounding_box
            self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())

        self.win.post_redraw()

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self):
        gui.Application.instance.run()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag",
                        default="/home/rosi/rosbags_96221/session_20260429_164113",
                        help="Path to one rosbag folder")
    parser.add_argument("--frame", type=int, default=10,
                        help="Frame index to preview (default: 10)")
    args = parser.parse_args()

    rclpy.init(args=None)
    print(f"Loading frame {args.frame} from {args.bag} ...")
    depth_msg, rgb_msg, cam_msg = load_frame(Path(args.bag), args.frame)
    rclpy.shutdown()

    z_min, z_max = load_config()
    print(f"Loaded config: z_min={z_min}  z_max={z_max}")

    app = RGBPointCloudApp(depth_msg, rgb_msg, cam_msg, z_min, z_max)
    app.run()


if __name__ == "__main__":
    main()
