#!/usr/bin/env python3
"""
compare_pointcloud.py
=====================
Captures one live depth frame from the ZED and displays it alongside
a reference frame from the training zarr.  Use this to verify the live
scene matches what the model was trained on.

Usage:
    python3 compare_pointcloud.py
    python3 compare_pointcloud.py --episode 5 --step 0   # specific training frame
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import yaml

import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo

ZARR_PATH    = "/home/rosi/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real_cable_pull.zarr"
CONFIG_PATH  = Path(__file__).parent / "z_filter_config.yaml"
BAGS_DIR     = Path("/home/rosi/rosbags_96221")


def load_z_filter():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return float(cfg.get("z_min", 0.0)), float(cfg.get("z_max", 9999.0))
    return 0.0, 9999.0


def depth_to_xyz(depth_msg: Image, cam_msg: CameraInfo,
                  z_min: float, z_max: float) -> np.ndarray:
    h, w = depth_msg.height, depth_msg.width
    depth = np.frombuffer(bytes(depth_msg.data), dtype=np.float32).reshape(h, w)
    fx, fy = cam_msg.k[0], cam_msg.k[4]
    cx, cy = cam_msg.k[2], cam_msg.k[5]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    pts = np.stack([(us-cx)*z/fx, (vs-cy)*z/fy, z], axis=-1).reshape(-1, 3)
    valid = np.isfinite(pts[:,2]) & (pts[:,2] >= z_min) & (pts[:,2] <= z_max)
    return pts[valid]


def load_live_frame() -> np.ndarray:
    """Grab one frame from the live ZED topics."""
    rclpy.init()
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    import threading, time

    class Grabber(Node):
        def __init__(self):
            super().__init__("pc_compare_grabber")
            be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE,
                            history=HistoryPolicy.KEEP_LAST, depth=1)
            self.depth = None; self.cam = None
            self.create_subscription(Image,      "/zed/zed_node/depth/depth_registered", self._d, be)
            self.create_subscription(CameraInfo, "/zed/zed_node/depth/camera_info",      self._c, be)
        def _d(self, m): self.depth = m
        def _c(self, m): self.cam   = m
        def ready(self): return self.depth and self.cam

    node = Grabber()
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()
    print("Waiting for ZED frame ...")
    for _ in range(100):
        if node.ready(): break
        time.sleep(0.05)
    if not node.ready():
        print("ERROR: no ZED data received — is the camera running?")
        rclpy.shutdown(); sys.exit(1)

    z_min, z_max = load_z_filter()
    pts = depth_to_xyz(node.depth, node.cam, z_min, z_max)
    rclpy.shutdown()
    print(f"Live frame: {len(pts):,} valid points  (z_min={z_min}, z_max={z_max})")
    return pts


def load_training_frame(episode: int, step: int) -> np.ndarray:
    """Load a raw point cloud frame from the training zarr."""
    from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
    buf = ReplayBuffer.copy_from_path(ZARR_PATH)
    ends = buf.episode_ends[:]
    start = 0 if episode == 0 else ends[episode - 1]
    end   = ends[episode]
    t = min(start + step, end - 1)
    pc = buf['point_cloud'][t]   # (1024, 3)
    # Remove zero-padded points
    valid = (pc != 0).any(axis=1)
    pts = pc[valid]
    print(f"Training ep={episode} step={step} (global={t}): {len(pts)} points")
    return pts


def make_o3d_pcd(pts: np.ndarray, color) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    colors = np.tile(color, (len(pts), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def show_side_by_side(live_pts: np.ndarray, train_pts: np.ndarray):
    app = gui.Application.instance
    app.initialize()

    win = app.create_window("Point Cloud Comparison  (left=Live  right=Training)", 1400, 700)
    em  = win.theme.font_size

    # Left — live (blue)
    left  = gui.SceneWidget()
    left.scene = rendering.Open3DScene(win.renderer)
    left.scene.set_background([0.1, 0.1, 0.1, 1.0])
    left.scene.show_axes(True)

    # Right — training (orange)
    right = gui.SceneWidget()
    right.scene = rendering.Open3DScene(win.renderer)
    right.scene.set_background([0.1, 0.1, 0.1, 1.0])
    right.scene.show_axes(True)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.0

    live_pcd  = make_o3d_pcd(live_pts,  [0.2, 0.6, 1.0])   # blue
    train_pcd = make_o3d_pcd(train_pts, [1.0, 0.5, 0.1])   # orange

    left.scene.add_geometry("live",  live_pcd,  mat)
    right.scene.add_geometry("train", train_pcd, mat)

    # Labels
    panel = gui.Vert(0, gui.Margins(em, em//2, em, em//2))
    panel.add_child(gui.Label(f"LEFT  (blue)  : Live ZED  —  {len(live_pts):,} pts"))
    panel.add_child(gui.Label(f"RIGHT (orange): Training sample  —  {len(train_pts):,} pts"))
    panel.add_child(gui.Label("Compare: are the cables in the same location and orientation?"))
    panel.add_fixed(em)
    panel.add_child(gui.Label("If live is mostly background with no cable geometry → move/place cables."))
    panel.add_child(gui.Label("If live looks similar to training → model distribution issue."))

    def on_layout(ctx):
        r   = win.content_rect
        pw  = 420
        ph  = 60
        hw  = (r.width - pw) // 2
        left.frame  = gui.Rect(r.x,          r.y + ph, hw,   r.height - ph)
        right.frame = gui.Rect(r.x + hw,     r.y + ph, hw,   r.height - ph)
        panel.frame = gui.Rect(r.x,          r.y,      r.width, ph)

    win.set_on_layout(on_layout)
    win.add_child(left)
    win.add_child(right)
    win.add_child(panel)

    # Auto-fit cameras
    for widget, pcd in [(left, live_pcd), (right, train_pcd)]:
        bb = pcd.get_axis_aligned_bounding_box()
        widget.setup_camera(60.0, bb, bb.get_center())

    win.set_on_close(lambda: (gui.Application.instance.quit(), True)[1])
    app.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0,  help="Training episode index")
    parser.add_argument("--step",    type=int, default=0,  help="Step within episode")
    parser.add_argument("--no-live", action="store_true",  help="Skip live capture, compare two training frames")
    args = parser.parse_args()

    train_pts = load_training_frame(args.episode, args.step)

    if args.no_live:
        train_pts2 = load_training_frame(args.episode, args.step + 5)
        show_side_by_side(train_pts2, train_pts)
    else:
        live_pts = load_live_frame()
        show_side_by_side(live_pts, train_pts)


if __name__ == "__main__":
    main()
