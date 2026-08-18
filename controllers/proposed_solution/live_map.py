"""
live_map.py
===========
Maintains an incremental occupancy grid updated in real-time from Lidar scans.
Starts with the static base map (walls), and marks new obstacle cells (beds,
furniture) as they are observed by the robot's Lidar.
"""

import json
import math
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view


# Maximum distance (in metres) to paint Lidar hits onto the map.
# Ignoring very far returns prevents noisy measurements from blocking open corridors.
MAX_PAINT_RANGE = 2.5
MIN_PAINT_RANGE = 0.12   # Ignore hits too close to robot body


class LiveMap:
    """
    Incremental 2D occupancy grid that fuses the static floor plan
    with real-time Lidar point returns.
    """

    def __init__(self, map_png_path: str, metadata_path: str):
        """
        Args:
            map_png_path  : Path to map_estimate.png (black walls, white free)
            metadata_path : Path to map_metadata.json (coordinate transform)
        """
        with open(metadata_path, "r") as f:
            meta = json.load(f)

        self._world_min_x = meta["world_min_x"]
        self._world_max_y = meta["world_max_y"]
        self._res         = meta["resolution"]     # metres per pixel
        self._width       = meta["width"]
        self._height      = meta["height"]
        self._origin_x    = meta["origin_x"]
        self._origin_y    = meta["origin_y"]

        # Base static map (True = occupied)
        img = Image.open(map_png_path).convert("L")
        arr = np.array(img, dtype=np.uint8)
        self._base_grid = arr < 128

        # Dynamic obstacles grid (starts empty)
        self._dynamic_grid = np.zeros_like(self._base_grid, dtype=bool)

        # Cache for inflated grid
        self._dirty = True
        self._cached_inflated_grid = None
        self._cached_inflation_px = None

    # ── Map update ───────────────────────────────────────────────────────────

    def update(self, odom_x: float, odom_y: float, odom_heading: float, lidar) -> int:
        """
        Read Lidar range image, clear free space along each ray, and mark hit endpoints.

        Args:
            odom_x, odom_y : Origin-relative robot position
            odom_heading   : Robot yaw heading (rad)
            lidar          : Webots Lidar device

        Returns:
            Current count of dynamic obstacle cells.
        """
        if lidar is None:
            return int(self._dynamic_grid.sum())

        ranges = lidar.getRangeImage()
        if not ranges:
            return int(self._dynamic_grid.sum())

        fov = lidar.getFov()
        n_pts = len(ranges)

        # Robot position in absolute world coordinates & pixel space
        rx = odom_x + self._origin_x
        ry = odom_y + self._origin_y
        r0 = int((self._world_max_y - ry) / self._res)
        c0 = int((rx - self._world_min_x) / self._res)

        if not (0 <= r0 < self._height and 0 <= c0 < self._width):
            return int(self._dynamic_grid.sum())

        # Step by 2 or 4 to process 100-200 rays quickly
        step = 2 if n_pts >= 200 else 1

        for i in range(0, n_pts, step):
            r = ranges[i]
            if math.isnan(r):
                continue

            angle = (i / n_pts) * fov
            if angle > math.pi:
                angle -= 2 * math.pi
            global_angle = odom_heading + angle

            is_hit = True
            if math.isinf(r) or r > MAX_PAINT_RANGE:
                r_trace = MAX_PAINT_RANGE
                is_hit = False
            elif r < MIN_PAINT_RANGE:
                continue
            else:
                r_trace = r

            wx = rx + r_trace * math.cos(global_angle)
            wy = ry + r_trace * math.sin(global_angle)

            r1 = int((self._world_max_y - wy) / self._res)
            c1 = int((wx - self._world_min_x) / self._res)

            r1 = max(0, min(self._height - 1, r1))
            c1 = max(0, min(self._width - 1, c1))

            # Bresenham raytrace to clear free space between (r0, c0) and (r1, c1)
            dr = abs(r1 - r0)
            dc = abs(c1 - c0)
            sr = 1 if r1 > r0 else -1
            sc = 1 if c1 > c0 else -1
            err = dr - dc

            curr_r, curr_c = r0, c0
            while curr_r != r1 or curr_c != c1:
                self._dynamic_grid[curr_r, curr_c] = False
                e2 = 2 * err
                if e2 > -dc:
                    err -= dc
                    curr_r += sr
                if e2 < dr:
                    err += dr
                    curr_c += sc

            # Mark endpoint as obstacle if it was a valid hit
            if is_hit:
                # Only mark if not already a base wall
                if not self._base_grid[r1, c1]:
                    self._dynamic_grid[r1, c1] = True

        self._dirty = True
        return int(self._dynamic_grid.sum())

    # ── Grid access & inflation ──────────────────────────────────────────────

    def get_inflated_grid(self, inflation_px: int = 5) -> np.ndarray:
        """
        Return the inflated combined (base + dynamic) occupancy grid for A* planning.
        Caches the result until new updates occur.
        """
        if (
            self._dirty
            or self._cached_inflated_grid is None
            or self._cached_inflation_px != inflation_px
        ):
            combined_grid = self._base_grid | self._dynamic_grid
            self._cached_inflated_grid = self._inflate(combined_grid, inflation_px)
            self._cached_inflation_px = inflation_px
            self._dirty = False

        return self._cached_inflated_grid

    @property
    def live_grid(self) -> np.ndarray:
        """Uninflated raw combined occupancy grid."""
        return self._base_grid | self._dynamic_grid

    @property
    def painted_cells_count(self) -> int:
        """Total number of currently active dynamic obstacle cells."""
        return int(self._dynamic_grid.sum())

    # ── Obstacle inflation ───────────────────────────────────────────────────

    @staticmethod
    def _inflate(grid: np.ndarray, radius_px: int) -> np.ndarray:
        """
        Expand every occupied cell by radius_px pixels (circular kernel).
        """
        if radius_px <= 0:
            return grid.copy()

        d = 2 * radius_px + 1
        y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
        kernel = (x * x + y * y <= radius_px * radius_px).astype(np.uint8)

        h, w = grid.shape
        padded = np.pad(grid.astype(np.uint8), radius_px, mode='edge')
        windows = sliding_window_view(padded, (d, d))
        inflated = (windows * kernel).max(axis=(2, 3)) > 0
        return inflated

    # ── Debug helpers ────────────────────────────────────────────────────────

    def save_debug_image(self, output_path: str):
        """Save a visual PNG of the live occupancy grid (black=walls/obstacles)."""
        img_arr = np.where(self._live_grid, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="L")
        img.save(output_path)
