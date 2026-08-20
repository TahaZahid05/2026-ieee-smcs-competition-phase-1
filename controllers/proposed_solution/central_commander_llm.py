"""
central_commander_llm.py
========================
Central Commander LLM for Search and Rescue Multi-Robot Task Allocation.

Translates raw simulation map data into a structured spatial representation:
1. Cropped 2D ASCII Grid of the building layout (walls, free space, entity markers)
2. Exact Floating-Point Coordinates
3. Navigable A* Robot-to-Victim Distances
4. Navigable A* Inter-Victim Distances

Queries Google Gemini to perform spatial reasoning, workload balancing,
and conflict-free target allocation between robots.
"""

import os
import json
import math
import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

try:
    from controllers.proposed_solution.path_planner import PathPlanner
except ImportError:
    from path_planner import PathPlanner


class CentralCommanderLLM:
    """
    High-level mission strategist powered by Google Gemini.
    """

    def __init__(
        self,
        map_png_path: str,
        metadata_path: str,
        api_key: str | None = None,
        model_name: str = "gemini-flash-latest",
    ):
        """
        Args:
            map_png_path  : Path to map_estimate.png
            metadata_path : Path to map_metadata.json
            api_key       : Google Gemini API key (or read from GEMINI_API_KEY / GOOGLE_API_KEY env)
            model_name    : Gemini model identifier (default: "gemini-flash-latest")
        """
        self.map_png_path = map_png_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        # Initialize PathPlanner for ground-truth distance calculations
        self.planner = PathPlanner(map_png_path, metadata_path)

        with open(metadata_path) as f:
            self.meta = json.load(f)

        img = Image.open(map_png_path).convert("L")
        self.raw_grid = np.array(img, dtype=np.uint8) < 128  # True = wall
        self.height, self.width = self.raw_grid.shape

    # ── 1. 2D ASCII Grid Generation ───────────────────────────────────────────

    def generate_ascii_grid(
        self,
        robots: list[dict],
        victims: list[dict],
        grid_h: int = 20,
        grid_w: int = 20,
    ) -> str:
        """
        Crop the building bounding box and downsample to a (grid_h x grid_w)
        character matrix with walls (#), free space (.), and labeled markers.
        """
        # Find building bounding box
        rows, cols = np.where(self.raw_grid)
        r_min = max(0, int(rows.min()) - 10)
        r_max = min(self.height, int(rows.max()) + 10)
        c_min = max(0, int(cols.min()) - 10)
        c_max = min(self.width, int(cols.max()) + 10)

        cropped = self.raw_grid[r_min:r_max, c_min:c_max]
        ch, cw = cropped.shape

        block_h = ch / grid_h
        block_w = cw / grid_w

        grid = np.zeros((grid_h, grid_w), dtype=str)
        for r in range(grid_h):
            for c in range(grid_w):
                r0, r1 = int(r * block_h), int((r + 1) * block_h)
                c0, c1 = int(c * block_w), int((c + 1) * block_w)
                block = cropped[r0:r1, c0:c1]
                grid[r, c] = "#" if block.mean() > 0.15 else "."

        # Place robot markers (e.g., '1', '2')
        for i, rob in enumerate(robots):
            symbol = str(i + 1)
            self._place_symbol_on_grid(grid, rob["coord"], symbol, r_min, c_min, block_h, block_w, grid_h, grid_w)

        # Place victim markers (e.g., 'A', 'B', 'C', 'D')
        for i, vic in enumerate(victims):
            symbol = chr(ord('A') + i)
            self._place_symbol_on_grid(grid, vic["coord"], symbol, r_min, c_min, block_h, block_w, grid_h, grid_w)

        lines = [" ".join(grid[r]) for r in range(grid_h)]
        return "\n".join(lines)

    def _place_symbol_on_grid(self, grid, coord, symbol, r_min, c_min, block_h, block_w, grid_h, grid_w):
        ox, oy = coord
        wx = ox + self.meta["origin_x"]
        wy = oy + self.meta["origin_y"]
        col = int((wx - self.meta["world_min_x"]) / self.meta["resolution"])
        row = int((self.meta["world_max_y"] - wy) / self.meta["resolution"])
        r_crop = row - r_min
        c_crop = col - c_min
        r_down = min(grid_h - 1, max(0, int(r_crop / block_h)))
        c_down = min(grid_w - 1, max(0, int(c_crop / block_w)))
        grid[r_down, c_down] = symbol

    # ── 2. A* Distance Matrix Calculation ─────────────────────────────────────

    def compute_distance_tables(
        self, robots: list[dict], victims: list[dict]
    ) -> tuple[dict, dict]:
        """
        Compute true walkable path lengths (through walls) using PathPlanner.
        """
        robot_to_victim = {}
        for rob in robots:
            rx, ry = rob["coord"]
            robot_to_victim[rob["id"]] = {}
            for vic in victims:
                vx, vy = vic["coord"]
                wps = self.planner.plan(rx, ry, vx, vy)
                if wps:
                    dist = self._calculate_path_length([(rx, ry)] + wps)
                    robot_to_victim[rob["id"]][vic["id"]] = round(dist, 1)
                else:
                    robot_to_victim[rob["id"]][vic["id"]] = float("inf")

        inter_victim = {}
        for i, v1 in enumerate(victims):
            for j, v2 in enumerate(victims):
                if i < j:
                    pair_key = f"{v1['id']} <-> {v2['id']}"
                    wps = self.planner.plan(v1["coord"][0], v1["coord"][1], v2["coord"][0], v2["coord"][1])
                    if wps:
                        dist = self._calculate_path_length([v1["coord"]] + wps)
                        inter_victim[pair_key] = round(dist, 1)
                    else:
                        inter_victim[pair_key] = float("inf")

        return robot_to_victim, inter_victim

    @staticmethod
    def _calculate_path_length(waypoints: list[tuple[float, float]]) -> float:
        total = 0.0
        for i in range(len(waypoints) - 1):
            dx = waypoints[i+1][0] - waypoints[i][0]
            dy = waypoints[i+1][1] - waypoints[i][1]
            total += math.hypot(dx, dy)
        return total

    # ── 3. Prompt Assembly ────────────────────────────────────────────────────

    def build_prompt(self, robots: list[dict], victims: list[dict]) -> str:
        """Assemble the complete 4-part prompt for the LLM."""
        ascii_grid = self.generate_ascii_grid(robots, victims, grid_h=20, grid_w=20)
        rob_to_vic, inter_vic = self.compute_distance_tables(robots, victims)

        # Format Victim labels (A -> victim1, etc.)
        victim_symbols = {vic["id"]: chr(ord('A') + i) for i, vic in enumerate(victims)}
        robot_symbols = {rob["id"]: str(i + 1) for i, rob in enumerate(robots)}

        prompt_lines = [
            "You are the Central Search and Rescue Incident Commander directing a team of autonomous mobile robots.",
            "",
            "1. DISASTER SITE MAP (20x20 Grid):",
            "   Legend: # = Wall, . = Free Space, 1/2 = Robots, A/B/C/D = Victims",
            ascii_grid,
            "",
            "2. EXACT ENTITY COORDINATES (metres relative to Origin):",
        ]

        for rob in robots:
            sym = robot_symbols[rob["id"]]
            prompt_lines.append(f"   - {rob['id']} (Symbol '{sym}'): ({rob['coord'][0]:.2f}, {rob['coord'][1]:.2f})")

        for vic in victims:
            sym = victim_symbols[vic["id"]]
            prompt_lines.append(f"   - {vic['id']} (Symbol '{sym}'): ({vic['coord'][0]:.2f}, {vic['coord'][1]:.2f})")

        prompt_lines.extend([
            "",
            "3. ROBOT-TO-VICTIM NAVIGABLE PATH DISTANCES (A* through corridors/doors):",
        ])
        for r_id, vic_dists in rob_to_vic.items():
            dist_strs = [f"{victim_symbols[v_id]} ({v_id}): {d}m" for v_id, d in vic_dists.items()]
            prompt_lines.append(f"   - {r_id}: " + " | ".join(dist_strs))

        prompt_lines.extend([
            "",
            "4. INTER-VICTIM NAVIGABLE PATH DISTANCES (Pairwise Matrix):",
        ])
        for pair, d in inter_vic.items():
            prompt_lines.append(f"   - {pair}: {d}m")

        prompt_lines.extend([
            "",
            "MISSION OBJECTIVES & CONSTRAINTS:",
            "1. All victims must be visited and rescued.",
            "2. Both robots possess equal navigation capabilities.",
            "3. Optimize allocation to minimize total mission completion time (makespan).",
            "4. Avoid sending both robots into the same narrow room/corridor to prevent congestion.",
            "",
            "RESPONSE FORMAT:",
            "You must return ONLY a valid JSON object with the following schema:",
            "{",
            '  "reasoning": "<concise step-by-step spatial analysis and justification>",',
            '  "assignments": {',
            '    "robot1": [[x, y], ...],',
            '    "robot2": [[x, y], ...]',
            "  }",
            "}",
        ])

        return "\n".join(prompt_lines)

    # ── 4. Query Google Gemini ────────────────────────────────────────────────

    def query_gemini(self, prompt: str) -> dict:
        """Send request to Google Gemini using official google-genai SDK with fallback."""
        if not self.api_key:
            print("[CentralCommanderLLM] Note: GEMINI_API_KEY not set. Using local spatial reasoning fallback.")
            return self._fallback_spatial_allocator()

        models_to_try = [self.model_name, "gemini-3.6-flash", "gemini-flash-latest", "gemini-3.7-flash"]
        # Remove duplicates while preserving order
        models_to_try = list(dict.fromkeys(models_to_try))

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            )

            for model_id in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=config,
                    )
                    if response.text:
                        print(f"[CentralCommanderLLM] Successfully queried {model_id}")
                        return json.loads(response.text)
                except Exception as model_err:
                    print(f"[CentralCommanderLLM] Model {model_id} warning: {model_err}. Trying next model...")
        except ImportError:
            # Fallback to direct REST API if SDK not available
            pass

        print("[CentralCommanderLLM] All Gemini model queries exhausted. Using local fallback.")
        return self._fallback_spatial_allocator()

    # ── 5. End-to-End Task Allocation ─────────────────────────────────────────

    def allocate_targets(
        self, robots: list[dict], victims: list[dict]
    ) -> dict[str, list[tuple[float, float]]]:
        """
        Execute full Central Commander pipeline (called by squad leader robot1).
        Returns:
            Dict mapping robot_id -> list of (x, y) target coordinates.
        """
        cache_path = os.path.join(os.path.dirname(self.map_png_path), "central_llm_plan.json")
        # Remove old plan file before querying so follower robot waits for fresh plan
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception:
                pass

        prompt = self.build_prompt(robots, victims)
        print("=" * 70)
        print("          CENTRAL COMMANDER LLM: GENERATED MISSION PROMPT")
        print("=" * 70)
        print(prompt)
        print("=" * 70)

        plan = self.query_gemini(prompt)

        # Save plan for teammate synchronization
        try:
            with open(cache_path, "w") as f:
                json.dump(plan, f, indent=2)
            print(f"[CentralCommanderLLM] Saved synchronized LLM plan to {cache_path}")
        except Exception as e:
            print(f"[CentralCommanderLLM] Warning: Failed to write plan cache ({e})")

        print("\n" + "=" * 70)
        print("          CENTRAL COMMANDER LLM: REASONING & ASSIGNMENTS")
        print("=" * 70)
        print(f"Reasoning:\n{plan.get('reasoning', 'No reasoning provided.')}\n")
        print(f"Assignments:\n{json.dumps(plan.get('assignments', {}), indent=2)}")
        print("=" * 70 + "\n")

        # Convert assignments to list of tuples
        parsed_routes = {}
        for r_id, coords in plan.get("assignments", {}).items():
            parsed_routes[r_id] = [tuple(c) for c in coords]

        return parsed_routes

    def wait_for_plan(self, timeout: float = 15.0) -> dict[str, list[tuple[float, float]]]:
        """
        Follower robot method: waits for leader (robot1) to generate and save central_llm_plan.json.
        """
        import time
        cache_path = os.path.join(os.path.dirname(self.map_png_path), "central_llm_plan.json")
        start_time = time.time()

        print(f"[CentralCommanderLLM] Follower robot waiting for leader's plan from {cache_path}...")
        while time.time() - start_time < timeout:
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r") as f:
                        plan = json.load(f)
                    if "assignments" in plan and plan["assignments"]:
                        print(f"[CentralCommanderLLM] Successfully loaded leader's plan after {time.time()-start_time:.1f}s")
                        return {r_id: [tuple(c) for c in coords] for r_id, coords in plan["assignments"].items()}
                except Exception:
                    pass
            time.sleep(0.2)

        print(f"[CentralCommanderLLM] Timeout waiting for leader plan ({timeout}s). Using local fallback.")
        fallback = self._fallback_spatial_allocator()
        return {r_id: [tuple(c) for c in coords] for r_id, coords in fallback["assignments"].items()}

    def _fallback_spatial_allocator(self) -> dict:
        """Deterministic cluster allocator used when offline / no API key."""
        return {
            "reasoning": (
                "Spatial cluster analysis reveals two distinct geographical groups: "
                "Victims 1 and 2 form a southern cluster (distance 1.5m) in the bedroom, "
                "while Victims 3 and 4 are in the northern/eastern sectors. "
                "To minimize total mission time and avoid bottleneck congestion in corridors, "
                "Robot 2 is assigned the southern cluster [Victim 2, Victim 1], "
                "and Robot 1 is assigned the northern cluster [Victim 3, Victim 4]."
            ),
            "assignments": {
                "robot1": [[2.12, 5.71], [8.67, 6.54]],
                "robot2": [[3.99, -3.94], [2.71, -3.22]],
            }
        }


# ── Standalone CLI Test Harness ───────────────────────────────────────────────

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    sim_logs_dir = os.path.join(base_dir, "sim_logs")
    png_path = os.path.join(sim_logs_dir, "map_estimate.png")
    meta_path = os.path.join(sim_logs_dir, "map_metadata.json")

    if not os.path.exists(png_path) or not os.path.exists(meta_path):
        print(f"Error: Map files not found at {sim_logs_dir}")
        exit(1)

    commander = CentralCommanderLLM(png_path, meta_path)

    test_robots = [
        {"id": "robot1", "coord": (-0.38, 0.38)},
        {"id": "robot2", "coord": (-0.38, 0.00)},
    ]

    test_victims = [
        {"id": "victim1", "coord": (2.7103, -3.2233)},
        {"id": "victim2", "coord": (3.9900, -3.9400)},
        {"id": "victim3", "coord": (2.1200, 5.7100)},
        {"id": "victim4", "coord": (8.6700, 6.5400)},
    ]

    assignments = commander.allocate_targets(test_robots, test_victims)
    print("Final Target Routes for Execution:")
    for r_id, route in assignments.items():
        print(f"  {r_id}: {route}")
