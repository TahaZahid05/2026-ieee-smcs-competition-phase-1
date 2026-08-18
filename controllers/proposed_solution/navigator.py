"""
navigator.py
============
Drives the robot through a list of (x, y) origin-relative waypoints
using a proportional heading controller. Obstacle avoidance is handled
globally via live occupancy grid mapping and A* replanning.
"""

import math


BASE_SPEED   = 3.0    # rad/s — forward speed when well aligned with waypoint
MAX_SPEED    = 9.0    # rad/s — motor velocity cap
TURN_GAIN    = 4.0    # proportional gain for heading correction
ARRIVAL_DIST = 0.30   # metres — distance threshold to pop current waypoint


class Navigator:
    """
    Proportional waypoint follower.
    """

    def __init__(self, robot_id, odometry, set_speeds_fn):
        """
        Args:
            robot_id      : String identifier (e.g. "robot1", "robot2")
            odometry      : Odometry instance
            set_speeds_fn : Callable(left: float, right: float) -> None
        """
        self._robot_id   = robot_id
        self._odom       = odometry
        self._set_speeds = set_speeds_fn

        self._waypoints: list[tuple[float, float]] = []
        self._mode       = "idle"    # "tracking" | "arrived" | "idle"

    # ── Public API ───────────────────────────────────────────────────────────

    def set_waypoints(self, waypoints: list[tuple[float, float]]):
        """Load a new list of (x, y) origin-relative waypoints to follow."""
        self._waypoints = list(waypoints)
        self._mode = "tracking" if waypoints else "arrived"
        if waypoints:
            print(f"[{self._robot_id}/Navigator] Route loaded: {len(waypoints)} waypoints, "
                  f"first → ({waypoints[0][0]:.2f}, {waypoints[0][1]:.2f})")

    def step(self) -> str:
        """
        Call once per simulation timestep. Computes and applies motor commands.

        Returns:
            "tracking"  — driving toward current waypoint
            "arrived"   — all waypoints reached, motors stopped
            "idle"      — no waypoints loaded
        """
        if self._mode in ("arrived", "idle"):
            self._set_speeds(0.0, 0.0)
            return self._mode

        if not self._waypoints:
            self._mode = "arrived"
            self._set_speeds(0.0, 0.0)
            print(f"[{self._robot_id}/Navigator] All waypoints reached.")
            return "arrived"

        # ── Check arrival at current waypoint ─────────────────────────────
        wp_x, wp_y = self._waypoints[0]
        dist = self._odom.distance_to(wp_x, wp_y)

        if dist < ARRIVAL_DIST:
            self._waypoints.pop(0)
            if self._waypoints:
                nxt = self._waypoints[0]
                print(f"[{self._robot_id}/Navigator] Waypoint reached. "
                      f"pos=({self._odom.x:.2f},{self._odom.y:.2f}) "
                      f"Next → ({nxt[0]:.2f}, {nxt[1]:.2f})")
            else:
                print(f"[{self._robot_id}/Navigator] Final waypoint reached. "
                      f"pos=({self._odom.x:.2f},{self._odom.y:.2f})")
            return self.step()

        # ── Waypoint Tracking (Proportional Heading Control) ──────────────
        error = self._odom.heading_error_to(wp_x, wp_y)

        # Rotate in place on sharp turns (> 25° / 0.45 rad) before driving forward.
        # This prevents the robot from cutting corners into doorframes while turning.
        if abs(error) > 0.45:
            forward = 0.0
            turn    = TURN_GAIN * error
        else:
            forward = BASE_SPEED * math.cos(error)
            turn    = TURN_GAIN * error

        left  = _clamp(forward - turn, -MAX_SPEED, MAX_SPEED)
        right = _clamp(forward + turn, -MAX_SPEED, MAX_SPEED)
        self._set_speeds(left, right)
        self._mode = "tracking"
        return "tracking"

    @property
    def is_arrived(self) -> bool:
        return self._mode == "arrived"

    @property
    def current_target(self) -> tuple[float, float] | None:
        return self._waypoints[0] if self._waypoints else None


# ── Helper ───────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
