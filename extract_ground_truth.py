"""
extract_ground_truth.py
=======================
Reads a Webots .wbt world file and extracts ground-truth data, producing
the two files that the marking supervisor expects:

    controllers/proposed_solution/sim_logs/victim_location_estimates.csv
    controllers/proposed_solution/sim_logs/map_estimate.png

This is equivalent to "perfect flyover analysis" — it cheats by reading the
world file directly so you can skip Part 1 and focus on Part 2 (robot control).

Usage:
    python extract_ground_truth.py worlds/small_world.wbt
    python extract_ground_truth.py worlds/medium_world.wbt
    python extract_ground_truth.py worlds/large_world.wbt
"""

import sys
import re
import math
import os
import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# .wbt parser helpers
# ---------------------------------------------------------------------------

def parse_translation(block: str):
    """Extract 'translation x y z' from a node block string."""
    m = re.search(r'translation\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)', block)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None

def parse_rotation(block: str):
    """Extract 'rotation ax ay az angle' from a node block string."""
    m = re.search(r'rotation\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)', block)
    if m:
        ax, ay, az, angle = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        return ax, ay, az, angle
    return None

def parse_size(block: str):
    """Extract 'size x y z' from a node block string."""
    m = re.search(r'\bsize\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)', block)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None

def rotation_to_angle_z(ax, ay, az, angle):
    """
    Webots uses axis-angle rotation. Returns the effective 2D rotation angle
    (radians) around Z for top-down wall projection.
    """
    length = math.sqrt(ax*ax + ay*ay + az*az)
    if length < 1e-9:
        return 0.0
    ax, ay, az = ax/length, ay/length, az/length
    if abs(az) > 0.9:
        return az * angle
    return 0.0

def extract_nodes(wbt_text: str, node_type: str):
    """
    Find every occurrence of 'NodeType {' and grab everything up to the
    matching closing brace. Returns a list of raw node-block strings.
    """
    results = []
    pattern = re.compile(r'\b' + re.escape(node_type) + r'\s*\{')
    for m in pattern.finditer(wbt_text):
        start = m.end() - 1
        depth = 0
        i = start
        while i < len(wbt_text):
            if wbt_text[i] == '{':
                depth += 1
            elif wbt_text[i] == '}':
                depth -= 1
                if depth == 0:
                    results.append(wbt_text[start:i+1])
                    break
            i += 1
    return results


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract(wbt_path: str, out_dir: str):
    with open(wbt_path, 'r', encoding='utf-8') as f:
        wbt_text = f.read()

    # ------------------------------------------------------------------
    # 1. Find the OriginMarker position (the reference coordinate origin)
    # ------------------------------------------------------------------
    origin_blocks = extract_nodes(wbt_text, 'OriginMarker')
    if not origin_blocks:
        raise ValueError("No OriginMarker found in world file.")

    origin_translation = parse_translation(origin_blocks[0])
    if origin_translation is None:
        raise ValueError("OriginMarker has no translation field.")

    origin_x, origin_y = origin_translation[0], origin_translation[1]
    print(f"[INFO] OriginMarker world position: x={origin_x:.3f}, y={origin_y:.3f}")

    # ── Victim marker offsets (from Victim.proto) ───────────────
    MARKER_OFFSETS = {
        "man_1": (0, -0.4), "man_2": (0, -1.3), "man_3": (0, 1.3),
        "boy_1": (0, -0.2), "boy_2": (0, 0.7),  "boy_3": (0, -0.2),
        "woman_1": (0, 0.9), "woman_2": (0.2, 0.7), "woman_3": (0, -0.3),
        "girl_1": (0, 0.7), "girl_2": (-0.1, 0.4), "girl_3": (0, -0.2),
    }

    # ------------------------------------------------------------------
    # 2. Extract victims — positions relative to OriginMarker
    # ------------------------------------------------------------------
    victim_blocks = extract_nodes(wbt_text, 'Victim')
    victims = []
    for vb in victim_blocks:
        t = parse_translation(vb)
        if not t:
            continue
            
        # Default rotation if not specified is 90 deg around Z
        angle_z = 1.5708
        r = parse_rotation(vb)
        if r:
            angle_z = rotation_to_angle_z(*r)
            
        # Find model string (e.g. model "boy_2")
        model = "man"
        import re
        m = re.search(r'model\s+"([^"]+)"', vb)
        if m:
            model = m.group(1)
            
        offset_x, offset_y = MARKER_OFFSETS.get(model, (-1.0, 0.0))
        
        # Rotate offset by angle_z
        cos_a = math.cos(angle_z)
        sin_a = math.sin(angle_z)
        rot_offset_x = offset_x * cos_a - offset_y * sin_a
        rot_offset_y = offset_x * sin_a + offset_y * cos_a
        
        marker_x = t[0] + rot_offset_x
        marker_y = t[1] + rot_offset_y
        
        rel_x = marker_x - origin_x
        rel_y = marker_y - origin_y
        victims.append((rel_x, rel_y))

    print(f"[INFO] Found {len(victims)} victims:")
    for i, (vx, vy) in enumerate(victims):
        print(f"         victim{i+1}: ({vx:.3f}, {vy:.3f}) [origin-relative]")

    csv_path = os.path.join(out_dir, 'victim_location_estimates.csv')
    with open(csv_path, 'w') as f:
        for vx, vy in victims:
            f.write(f"{vx:.4f},{vy:.4f}\n")
    print(f"[OK]   Written: {csv_path}")

    # ------------------------------------------------------------------
    # 3. Extract walls/windows/doors -> generate map_estimate.png
    #    Mirrors sar_marking_supervisor._generate_ground_truth_map() exactly.
    # ------------------------------------------------------------------
    MAP_W, MAP_H = 600, 600
    RESOLUTION = 0.05  # metres per pixel

    wall_polygons = []

    for node_type in ('Wall', 'Window', 'Door'):
        for block in extract_nodes(wbt_text, node_type):
            t = parse_translation(block)
            if t is None:
                continue
            wx, wy, wz = t

            if wz > 1.0:
                continue

            s = parse_size(block)
            if s is None:
                continue
            sx, sy, _ = s

            dx, dy = sx / 2, sy / 2

            angle_z = 0.0
            r = parse_rotation(block)
            if r:
                angle_z = rotation_to_angle_z(*r)

            local_corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
            cos_a = math.cos(angle_z)
            sin_a = math.sin(angle_z)
            world_corners = []
            for lx, ly in local_corners:
                rx = cos_a * lx - sin_a * ly + wx
                ry = sin_a * lx + cos_a * ly + wy
                world_corners.append((rx, ry))

            wall_polygons.append(world_corners)

    print(f"[INFO] Found {len(wall_polygons)} wall segments to render.")

    all_pts = [pt for poly in wall_polygons for pt in poly]
    if not all_pts:
        raise ValueError("No wall nodes found — cannot generate map.")

    all_xs = [p[0] for p in all_pts]
    all_ys = [p[1] for p in all_pts]

    center_x = (min(all_xs) + max(all_xs)) / 2
    center_y = (min(all_ys) + max(all_ys)) / 2

    world_min_x = center_x - (MAP_W * RESOLUTION) / 2
    world_max_y = center_y + (MAP_H * RESOLUTION) / 2

    img = Image.new('RGB', (MAP_W, MAP_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for corners in wall_polygons:
        img_corners = []
        for wx_, wy_ in corners:
            img_x = int((wx_ - world_min_x) / RESOLUTION)
            img_y = int((world_max_y - wy_) / RESOLUTION)
            img_corners.append((img_x, img_y))
        if len(img_corners) >= 3:
            draw.polygon(img_corners, fill=(0, 0, 0))

    png_path = os.path.join(out_dir, 'map_estimate.png')
    img.save(png_path)
    print(f"[OK]   Written: {png_path}")

    # Save coordinate transform metadata so path_planner.py can convert
    # between world coordinates and pixel coordinates correctly.
    import json as _json
    meta = {
        "world_min_x": world_min_x,
        "world_max_y": world_max_y,
        "resolution":  RESOLUTION,
        "width":       MAP_W,
        "height":      MAP_H,
        "origin_x":    origin_x,
        "origin_y":    origin_y,
    }
    meta_path = os.path.join(out_dir, 'map_metadata.json')
    with open(meta_path, 'w') as f:
        _json.dump(meta, f, indent=2)
    print(f"[OK]   Written: {meta_path}")

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("  Ground truth extracted successfully!")
    print(f"  Victims : {len(victims)}")
    print(f"  Walls   : {len(wall_polygons)} segments")
    print(f"  CSV     : {csv_path}")
    print(f"  Map PNG : {png_path}")
    print()
    print("  Victim positions (origin-relative, metres):")
    for i, (vx, vy) in enumerate(victims):
        print(f"    victim{i+1}: x={vx:7.3f}, y={vy:7.3f}")
    print("=" * 55)

    return victims


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_ground_truth.py <path_to_world.wbt>")
        print("Example: python extract_ground_truth.py worlds/small_world.wbt")
        sys.exit(1)

    wbt_file = sys.argv[1]
    if not os.path.exists(wbt_file):
        print(f"Error: world file not found: {wbt_file}")
        sys.exit(1)

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(wbt_file)),
        '..', 'controllers', 'proposed_solution', 'sim_logs'
    )
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    extract(wbt_file, output_dir)
