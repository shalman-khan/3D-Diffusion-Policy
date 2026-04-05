#!/usr/bin/env python3
"""
pc_filter_viewer.py
===================
Interactive point cloud workspace filter for DP3 zarr datasets.

Shows a 3D Open3D viewer with sliders to define a cuboid workspace filter.
Points inside the cuboid are shown in colour; outside points are shown in grey.
On Save, writes:
  - filter_config.yaml  (the cuboid bounds)
  - a new filtered zarr  (re-FPS to original N after crop, state/action unchanged)

Usage:
  python scripts/pc_filter_viewer.py --zarr data/two_arm_lift_demo.zarr
  python scripts/pc_filter_viewer.py --zarr data/two_arm_lift_demo.zarr \\
      --out-zarr data/two_arm_lift_filtered.zarr \\
      --config filter_config.yaml \\
      --sample-episode 0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import zarr
import yaml
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from numcodecs import Blosc


# ---------------------------------------------------------------------------
# FPS (same as convert_rosbags_to_zarr, no torch dependency)
# ---------------------------------------------------------------------------

def fps_numpy(points: np.ndarray, n_samples: int, pre_subsample: int = 8192) -> np.ndarray:
    n = len(points)
    if n == 0:
        return np.zeros((n_samples, 3), dtype=np.float32)
    if n <= n_samples:
        idx = np.random.choice(n, n_samples, replace=True)
        return points[idx].astype(np.float32)
    if n > pre_subsample:
        idx = np.random.choice(n, pre_subsample, replace=False)
        points = points[idx]
        n = pre_subsample
    selected = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(n, np.inf, dtype=np.float64)
    selected[0] = np.random.randint(n)
    for i in range(1, n_samples):
        last = points[selected[i - 1]].astype(np.float64)
        d = np.sum((points.astype(np.float64) - last) ** 2, axis=1)
        distances = np.minimum(distances, d)
        selected[i] = np.argmax(distances)
    return points[selected].astype(np.float32)


# ---------------------------------------------------------------------------
# Zarr helpers
# ---------------------------------------------------------------------------

def load_episode_pc(zarr_path: str, episode_idx: int) -> np.ndarray:
    """Load all point cloud frames for one episode, return (T, N, 3)."""
    z = zarr.open(zarr_path, "r")
    ends = z["meta/episode_ends"][:]
    start = 0 if episode_idx == 0 else ends[episode_idx - 1]
    end   = ends[episode_idx]
    return z["data/point_cloud"][start:end]   # (T, 1024, 3)


def get_n_episodes(zarr_path: str) -> int:
    z = zarr.open(zarr_path, "r")
    return len(z["meta/episode_ends"])


def apply_filter_and_save(zarr_path: str, out_zarr_path: str,
                          bounds: dict, config_path: str):
    """
    Apply cuboid filter to every frame in zarr, re-FPS to original N,
    write new zarr. state and action are copied unchanged.
    """
    z_in = zarr.open(zarr_path, "r")
    pc_all   = z_in["data/point_cloud"][:]   # (T, N, 3)
    state    = z_in["data/state"][:]
    action   = z_in["data/action"][:]
    ep_ends  = z_in["meta/episode_ends"][:]

    T, N, _ = pc_all.shape
    n_pts = N  # keep same resolution

    print(f"Applying filter to {T} frames (re-FPS to {n_pts} pts each)...")

    xmin, xmax = bounds["x_min"], bounds["x_max"]
    ymin, ymax = bounds["y_min"], bounds["y_max"]
    zmin, zmax = bounds["z_min"], bounds["z_max"]

    filtered_pc = np.zeros((T, n_pts, 3), dtype=np.float32)

    for i in range(T):
        pts = pc_all[i]   # (N, 3)
        mask = (
            (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
            (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
            (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
        )
        inside = pts[mask]
        filtered_pc[i] = fps_numpy(inside, n_pts)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{T} frames done")

    # Save filter config
    with open(config_path, "w") as f:
        yaml.dump({"workspace": bounds}, f, default_flow_style=False)
    print(f"Filter config saved to {config_path}")

    # Write new zarr
    compressor = Blosc(cname="lz4", clevel=5)
    z_out = zarr.open(out_zarr_path, mode="w")
    z_out.require_group("data")
    z_out.require_group("meta")
    z_out["data"].array("point_cloud", filtered_pc,
                        chunks=(1, n_pts, 3), compressor=compressor, dtype="f4")
    z_out["data"].array("state",  state,  chunks=(1, state.shape[1]),  compressor=compressor, dtype="f4")
    z_out["data"].array("action", action, chunks=(1, action.shape[1]), compressor=compressor, dtype="f4")
    z_out["meta"].array("episode_ends", ep_ends, dtype="i8")
    print(f"Filtered zarr saved to {out_zarr_path}")


# ---------------------------------------------------------------------------
# Viewer app
# ---------------------------------------------------------------------------

class PCFilterApp:

    def __init__(self, zarr_path: str, out_zarr_path: str,
                 config_path: str, episode_idx: int):
        self.zarr_path    = zarr_path
        self.out_zarr_path = out_zarr_path
        self.config_path  = config_path
        self.episode_idx  = episode_idx
        self.n_episodes   = get_n_episodes(zarr_path)

        # Load all frames for selected episode, flatten to (T*N, 3)
        print(f"Loading episode {episode_idx} from {zarr_path} ...")
        ep_pc = load_episode_pc(zarr_path, episode_idx)  # (T, N, 3)
        # Use middle frame as representative view; also keep first/last for context
        mid = len(ep_pc) // 2
        sample_frames = np.concatenate([ep_pc[0], ep_pc[mid], ep_pc[-1]], axis=0)
        self.all_pts = sample_frames  # (3*N, 3)

        # Compute data range for slider limits
        self.data_bounds = {
            "x_min": float(self.all_pts[:, 0].min()),
            "x_max": float(self.all_pts[:, 0].max()),
            "y_min": float(self.all_pts[:, 1].min()),
            "y_max": float(self.all_pts[:, 1].max()),
            "z_min": float(self.all_pts[:, 2].min()),
            "z_max": float(self.all_pts[:, 2].max()),
        }

        # Current filter bounds (start = full range)
        self.bounds = dict(self.data_bounds)

        self._build_ui()

    def _build_ui(self):
        app = gui.Application.instance
        app.initialize()

        self.window = app.create_window("PC Filter Viewer", 1200, 800)
        w = self.window

        em = w.theme.font_size
        margin = gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em)

        # 3D scene panel
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(w.renderer)
        self.scene.scene.set_background([0.15, 0.15, 0.15, 1.0])

        # Right panel
        panel = gui.Vert(0, margin)

        # Episode selector
        ep_label = gui.Label(f"Episode (0–{self.n_episodes-1})")
        self.ep_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self.ep_edit.int_value = self.episode_idx
        self.ep_edit.set_limits(0, self.n_episodes - 1)
        self.ep_edit.set_on_value_changed(self._on_episode_change)
        panel.add_child(ep_label)
        panel.add_child(self.ep_edit)
        panel.add_fixed(em)

        # Sliders for each axis
        db = self.data_bounds
        self.sliders = {}
        slider_specs = [
            ("x_min", db["x_min"], db["x_max"]),
            ("x_max", db["x_min"], db["x_max"]),
            ("y_min", db["y_min"], db["y_max"]),
            ("y_max", db["y_min"], db["y_max"]),
            ("z_min", db["z_min"], db["z_max"]),
            ("z_max", db["z_min"], db["z_max"]),
        ]
        for name, lo, hi in slider_specs:
            lbl = gui.Label(name.replace("_", " "))
            sl  = gui.Slider(gui.Slider.DOUBLE)
            sl.set_limits(lo, hi)
            sl.double_value = self.bounds[name]
            sl.set_on_value_changed(lambda val, n=name: self._on_slider(n, val))
            self.sliders[name] = sl
            panel.add_child(lbl)
            panel.add_child(sl)

        panel.add_fixed(em)

        # Info label
        self.info_label = gui.Label("Points inside: –")
        panel.add_child(self.info_label)
        panel.add_fixed(em)

        # Buttons
        save_btn = gui.Button("Save Config + Write Filtered Zarr")
        save_btn.set_on_clicked(self._on_save)
        panel.add_child(save_btn)

        panel.add_fixed(0.5 * em)
        reset_btn = gui.Button("Reset to Full Range")
        reset_btn.set_on_clicked(self._on_reset)
        panel.add_child(reset_btn)

        # Layout: scene left, panel right
        layout = gui.Horiz(0, gui.Margins(0, 0, 0, 0))
        layout.add_child(self.scene)
        layout.add_fixed(0.5 * em)
        layout.add_child(panel)

        w.add_child(layout)
        w.set_on_layout(self._on_layout)

        self._update_scene()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_w = 260
        self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)
        panel_x = r.x + r.width - panel_w
        # find panel widget and set its frame
        children = self.window.get_children()
        for c in children:
            if isinstance(c, gui.Horiz):
                horiz = c
                break
        horiz.frame = gui.Rect(r.x, r.y, r.width, r.height)

    def _on_slider(self, name: str, val: float):
        self.bounds[name] = val
        # Clamp: min <= max
        if name.endswith("_min") and self.bounds[name] > self.bounds[name.replace("min", "max")]:
            self.bounds[name] = self.bounds[name.replace("min", "max")]
            self.sliders[name].double_value = self.bounds[name]
        if name.endswith("_max") and self.bounds[name] < self.bounds[name.replace("max", "min")]:
            self.bounds[name] = self.bounds[name.replace("max", "min")]
            self.sliders[name].double_value = self.bounds[name]
        self._update_scene()

    def _on_episode_change(self, val):
        self.episode_idx = int(val)
        ep_pc = load_episode_pc(self.zarr_path, self.episode_idx)
        mid = len(ep_pc) // 2
        self.all_pts = np.concatenate([ep_pc[0], ep_pc[mid], ep_pc[-1]], axis=0)
        self._update_scene()

    def _on_reset(self):
        self.bounds = dict(self.data_bounds)
        for name, sl in self.sliders.items():
            sl.double_value = self.bounds[name]
        self._update_scene()

    def _on_save(self):
        apply_filter_and_save(
            self.zarr_path, self.out_zarr_path,
            self.bounds, self.config_path
        )
        dlg = gui.Dialog("Done")
        em = self.window.theme.font_size
        layout = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        layout.add_child(gui.Label(f"Saved:\n  {self.config_path}\n  {self.out_zarr_path}"))
        ok = gui.Button("OK")
        ok.set_on_clicked(self.window.close_dialog)
        layout.add_child(ok)
        dlg.add_child(layout)
        self.window.show_dialog(dlg)

    def _update_scene(self):
        pts = self.all_pts
        b   = self.bounds

        inside_mask = (
            (pts[:, 0] >= b["x_min"]) & (pts[:, 0] <= b["x_max"]) &
            (pts[:, 1] >= b["y_min"]) & (pts[:, 1] <= b["y_max"]) &
            (pts[:, 2] >= b["z_min"]) & (pts[:, 2] <= b["z_max"])
        )

        n_inside = int(inside_mask.sum())
        self.info_label.text = f"Points inside: {n_inside} / {len(pts)}"

        # Build coloured point cloud
        colours = np.full((len(pts), 3), [0.35, 0.35, 0.35])  # grey = outside

        # Colour inside points by height (z) using viridis-like gradient
        if n_inside > 0:
            z_vals = pts[inside_mask, 2]
            z_range = z_vals.max() - z_vals.min()
            if z_range < 1e-6:
                z_norm = np.zeros(n_inside)
            else:
                z_norm = (z_vals - z_vals.min()) / z_range
            # Simple blue→green→yellow gradient
            r = np.clip(1.5 * z_norm - 0.5, 0, 1)
            g = np.clip(1.5 * z_norm,       0, 1)
            bv = np.clip(1.0 - 1.5 * z_norm, 0, 1)
            colours[inside_mask] = np.stack([r, g, bv], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colours)

        # Wireframe cuboid
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[b["x_min"], b["y_min"], b["z_min"]],
            max_bound=[b["x_max"], b["y_max"], b["z_max"]]
        )
        bbox.color = [1.0, 0.8, 0.0]  # yellow box

        sc = self.scene.scene
        sc.clear_geometry()
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 2.0
        sc.add_geometry("pc", pcd, mat)

        line_mat = rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 2.0
        sc.add_geometry("bbox", bbox.get_box_points_and_lines()[0] if False else
                        _bbox_to_lineset(bbox), line_mat)

        # Set camera on first update
        if not hasattr(self, "_camera_set"):
            bounds_o3d = pcd.get_axis_aligned_bounding_box()
            self.scene.setup_camera(60, bounds_o3d, bounds_o3d.get_center())
            self._camera_set = True

    def run(self):
        gui.Application.instance.run()


def _bbox_to_lineset(bbox: o3d.geometry.AxisAlignedBoundingBox):
    """Convert AABB to LineSet for rendering."""
    mn = bbox.min_bound
    mx = bbox.max_bound
    pts = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ])
    lines = [[0,1],[1,2],[2,3],[3,0],
             [4,5],[5,6],[6,7],[7,4],
             [0,4],[1,5],[2,6],[3,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[1.0, 0.8, 0.0]] * len(lines))
    return ls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive point cloud workspace filter viewer")
    parser.add_argument("--zarr",    required=True,
                        help="Input zarr path (e.g. data/two_arm_lift_demo.zarr)")
    parser.add_argument("--out-zarr", default=None,
                        help="Output filtered zarr path (default: <input>_filtered.zarr)")
    parser.add_argument("--config",  default=None,
                        help="Output filter config YAML path (default: filter_config.yaml next to input zarr)")
    parser.add_argument("--sample-episode", type=int, default=0,
                        help="Episode index to preview in viewer (default: 0)")
    args = parser.parse_args()

    zarr_path = Path(args.zarr)
    out_zarr  = Path(args.out_zarr)  if args.out_zarr else zarr_path.parent / (zarr_path.stem + "_filtered.zarr")
    cfg_path  = Path(args.config)    if args.config   else zarr_path.parent / "filter_config.yaml"

    print(f"Input zarr:      {zarr_path}")
    print(f"Output zarr:     {out_zarr}")
    print(f"Filter config:   {cfg_path}")
    print(f"Preview episode: {args.sample_episode}")

    gui.Application.instance.initialize()
    app = PCFilterApp(str(zarr_path), str(out_zarr), str(cfg_path), args.sample_episode)
    app.run()


if __name__ == "__main__":
    main()
