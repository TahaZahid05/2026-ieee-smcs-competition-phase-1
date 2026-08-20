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
MIN_PAINT_RANGE = 0.06   # Ignore hits too close to sensor center (laser minRange is 0.05m)


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

        # Filter out points directly touching static walls to avoid double-inflation in corridors
        self._near_base_wall = self._inflate(self._base_grid, radius_px=1)

        # Dynamic obstacles confidence grid (0-12) & binary active mask
        self._dynamic_confidence = np.zeros_like(self._base_grid, dtype=np.int16)
        self._dynamic_grid = np.zeros_like(self._base_grid, dtype=bool)

        # Precompute circular kernel offsets for 4px (0.20m) dynamic inflation radius
        self._inflation_px = 4
        self._circle_offsets = [
            (y, x)
            for y in range(-self._inflation_px, self._inflation_px + 1)
            for x in range(-self._inflation_px, self._inflation_px + 1)
            if y * y + x * x <= self._inflation_px * self._inflation_px
        ]

        # Pre-inflate base static map ONCE at startup (5px = 0.25m for walls)
        self._base_inflated_grid = self._inflate(self._base_grid, radius_px=5)

    # ── Map update ───────────────────────────────────────────────────────────

    def update(self, odom_x: float, odom_y: float, odom_heading: float, lidar) -> int:
        """
        Read Lidar point cloud (or range image fallback), update persistent dynamic obstacle confidence,
        and clear free space along ray paths.

        Args:
            odom_x, odom_y : Origin-relative robot position
            odom_heading   : Robot yaw heading (rad)
            lidar          : Webots Lidar device

        Returns:
            Current count of dynamic obstacle cells.
        """
        if lidar is None:
            return int(self._dynamic_grid.sum())

        # Robot position in absolute world coordinates & pixel space
        rx = odom_x + self._origin_x
        ry = odom_y + self._origin_y
        r0 = int((self._world_max_y - ry) / self._res)
        c0 = int((rx - self._world_min_x) / self._res)

        if not (0 <= r0 < self._height and 0 <= c0 < self._width):
            return int(self._dynamic_grid.sum())

        cos_h = math.cos(odom_heading)
        sin_h = math.sin(odom_heading)

        # 1. Prefer native Webots Point Cloud (exact 3D Cartesian coordinates in robot body frame)
        points = None
        try:
            points = lidar.getPointCloud()
        except Exception:
            points = None

        if points:
            n_pts = len(points)
            step = 2 if n_pts >= 200 else 1
            for i in range(0, n_pts, step):
                pt = points[i]
                px = pt.x
                py = pt.y
                if math.isnan(px) or math.isnan(py):
                    continue

                dist = math.hypot(px, py)
                if dist < MIN_PAINT_RANGE:
                    continue

                is_hit = True
                if math.isinf(dist) or dist > MAX_PAINT_RANGE:
                    if math.isinf(dist) or dist <= 1e-6:
                        continue
                    # Trace clear ray up to MAX_PAINT_RANGE
                    scale = MAX_PAINT_RANGE / dist
                    px *= scale
                    py *= scale
                    is_hit = False

                # Transform from robot local frame (x forward, y left) to world coordinates
                wx = rx + px * cos_h - py * sin_h
                wy = ry + px * sin_h + py * cos_h

                r1 = int((self._world_max_y - wy) / self._res)
                c1 = int((wx - self._world_min_x) / self._res)

                r1 = max(0, min(self._height - 1, r1))
                c1 = max(0, min(self._width - 1, c1))

                # Bresenham raytrace to decrement confidence along beam
                dr = abs(r1 - r0)
                dc = abs(c1 - c0)
                sr = 1 if r1 > r0 else -1
                sc = 1 if c1 > c0 else -1
                err = dr - dc

                curr_r, curr_c = r0, c0
                while curr_r != r1 or curr_c != c1:
                    if self._dynamic_confidence[curr_r, curr_c] > 0:
                        self._dynamic_confidence[curr_r, curr_c] -= 1
                    e2 = 2 * err
                    if e2 > -dc:
                        err -= dc
                        curr_r += sr
                    if e2 < dr:
                        err += dr
                        curr_c += sc

                # Increment confidence on endpoint if valid hit and not touching static walls
                if is_hit:
                    if not self._near_base_wall[r1, c1]:
                        self._dynamic_confidence[r1, c1] = min(12, self._dynamic_confidence[r1, c1] + 4)

        else:
            # Fallback: Range image with verified Webots angle formula (left = +fov/2, right = -fov/2)
            ranges = lidar.getRangeImage()
            if not ranges:
                return int(self._dynamic_grid.sum())

            fov = lidar.getFov()
            n_pts = len(ranges)
            step = 2 if n_pts >= 200 else 1

            for i in range(0, n_pts, step):
                r = ranges[i]
                if math.isnan(r):
                    continue

                # In Webots, index 0 is LEFT (+fov/2), decreasing to RIGHT (-fov/2)
                angle = fov / 2.0 - (i / n_pts) * fov
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

                dr = abs(r1 - r0)
                dc = abs(c1 - c0)
                sr = 1 if r1 > r0 else -1
                sc = 1 if c1 > c0 else -1
                err = dr - dc

                curr_r, curr_c = r0, c0
                while curr_r != r1 or curr_c != c1:
                    if self._dynamic_confidence[curr_r, curr_c] > 0:
                        self._dynamic_confidence[curr_r, curr_c] -= 1
                    e2 = 2 * err
                    if e2 > -dc:
                        err -= dc
                        curr_r += sr
                    if e2 < dr:
                        err += dr
                        curr_c += sc

                if is_hit:
                    if not self._near_base_wall[r1, c1]:
                        self._dynamic_confidence[r1, c1] = min(12, self._dynamic_confidence[r1, c1] + 4)

        # Active dynamic obstacles are cells with confirmed confidence
        self._dynamic_grid = self._dynamic_confidence >= 2
        return int(self._dynamic_grid.sum())

    # ── Grid access & inflation ──────────────────────────────────────────────

    def get_inflated_grid(self, inflation_px: int | None = None) -> np.ndarray:
        """
        Fast on-the-fly combination of pre-inflated static walls and active dynamic obstacle points.
        Guarantees zero ghost obstacles when moving robots/objects clear out of view.
        """
        grid = self._base_inflated_grid.copy()
        active_r, active_c = np.where(self._dynamic_grid)
        for r, c in zip(active_r, active_c):
            for dy, dx in self._circle_offsets:
                ny, nx = r + dy, c + dx
                if 0 <= ny < self._height and 0 <= nx < self._width:
                    grid[ny, nx] = True
        return grid

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

    def save_debug_image(
        self,
        output_path: str,
        robot_x: float | None = None,
        robot_y: float | None = None,
        robot_heading: float | None = None,
        waypoints: list[tuple[float, float]] | None = None,
        target_pos: tuple[float, float] | None = None,
    ):
        """
        Render and save a rich RGB visual perception map:
        - White       : Free space
        - Dark Gray   : Base blueprint walls
        - Red         : Dynamic Lidar hits (armchairs, cabinets, tables)
        - Light Orange: Inflated safety buffer
        - Green       : Robot position & heading
        - Blue/Cyan   : Active A* path waypoints
        - Magenta     : Current target location
        """
        from PIL import ImageDraw

        h, w = self._height, self._width
        rgb_arr = np.full((h, w, 3), 255, dtype=np.uint8)

        # 1. Base walls -> Dark Gray
        rgb_arr[self._base_grid] = [60, 60, 60]

        # 2. Inflated buffer (base + dynamic) -> Light Yellow/Orange
        inflated = self.get_inflated_grid()
        buffer_mask = inflated & ~self._base_grid & ~self._dynamic_grid
        rgb_arr[buffer_mask] = [255, 230, 160]

        # 3. Dynamic obstacles (furniture / lidar hits) -> Red
        rgb_arr[self._dynamic_grid] = [230, 30, 30]

        img = Image.fromarray(rgb_arr, mode="RGB")
        draw = ImageDraw.Draw(img)

        def to_px(ox, oy):
            wx = ox + self._origin_x
            wy = oy + self._origin_y
            col = int((wx - self._world_min_x) / self._res)
            row = int((self._world_max_y - wy) / self._res)
            return max(0, min(w - 1, col)), max(0, min(h - 1, row))

        # 4. Draw target location (Magenta cross)
        if target_pos is not None:
            tc, tr = to_px(target_pos[0], target_pos[1])
            draw.line([(tc - 5, tr), (tc + 5, tr)], fill=(255, 0, 200), width=2)
            draw.line([(tc, tr - 5), (tc, tr + 5)], fill=(255, 0, 200), width=2)

        # 5. Draw active A* path waypoints (Blue line + Cyan dots)
        if waypoints and len(waypoints) > 0:
            px_wps = []
            if robot_x is not None and robot_y is not None:
                px_wps.append(to_px(robot_x, robot_y))
            for wx, wy in waypoints:
                px_wps.append(to_px(wx, wy))

            if len(px_wps) >= 2:
                draw.line(px_wps, fill=(0, 120, 255), width=2)

            for col, row in px_wps[1:]:
                draw.ellipse([col - 2, row - 2, col + 2, row + 2], fill=(0, 230, 255), outline=(0, 80, 200))

        # 6. Draw Robot pose (Green circle with heading pointer)
        if robot_x is not None and robot_y is not None:
            rc, rr = to_px(robot_x, robot_y)
            draw.ellipse([rc - 4, rr - 4, rc + 4, rr + 4], fill=(0, 220, 0), outline=(0, 100, 0))
            if robot_heading is not None:
                # Heading in Webots: 0 = +X (East), +pi/2 = +Y (North)
                # In pixel image: +X is right (+col), +Y is up (-row)
                dx = 8 * math.cos(robot_heading)
                dy = -8 * math.sin(robot_heading)
                draw.line([(rc, rr), (rc + dx, rr + dy)], fill=(0, 100, 0), width=2)

        img.save(output_path)
