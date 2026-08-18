"""
odometry.py
===========
Tracks the robot's (x, y, heading) position in origin-relative world coordinates.

Strategy:
  - Heading  : read directly from compass (absolute, no drift)
  - Position : dead-reckoning from wheel encoders, corrected by compass heading

This avoids the main failure mode of pure wheel odometry, where small heading errors
compound into large position errors over time.

Coordinate frame (matches extract_ground_truth.py):
  - Origin : OriginMarker position in the world file
  - X axis : East (right when looking down at the map)
  - Y axis : North (up when looking down at the map)
  - Heading: 0 = East (+X), increases counter-clockwise (standard math convention)
"""

import math


# ── Robot physical constants (from Rosbot.proto) ────────────────────────────
WHEEL_RADIUS = 0.043   # metres (from wheel cylinder radius)
WHEEL_BASE   = 0.22    # metres (from left y=0.11 to right y=-0.11)


class Odometry:
    """
    Incremental position estimator using wheel encoders + compass.

    Usage:
        odom = Odometry(compass, fl_sensor, fr_sensor,
                        initial_x=-0.375, initial_y=0.375)

        while robot.step(timestep) != -1:
            odom.update()
            print(odom.x, odom.y, math.degrees(odom.heading))
    """

    def __init__(
        self,
        compass,
        fl_sensor,
        fr_sensor,
        initial_x: float = 0.0,
        initial_y: float = 0.0,
    ):
        """
        Args:
            compass    : Webots Compass device (already enabled)
            fl_sensor  : Webots PositionSensor for front-left wheel (already enabled)
            fr_sensor  : Webots PositionSensor for front-right wheel (already enabled)
            initial_x  : Starting X position in origin-relative metres
            initial_y  : Starting Y position in origin-relative metres
        """
        self._compass  = compass
        self._fl       = fl_sensor
        self._fr       = fr_sensor

        self._x       = float(initial_x)
        self._y       = float(initial_y)
        self._heading = 0.0          # set on first update() call

        # Previous encoder readings — initialised on first call so we don't
        # get a spurious large delta from an undefined initial value.
        self._prev_fl: float | None = None
        self._prev_fr: float | None = None

        self._initialised = False

    # ── Public properties ────────────────────────────────────────────────────

    @property
    def x(self) -> float:
        """Current X position in origin-relative metres."""
        return self._x

    @property
    def y(self) -> float:
        """Current Y position in origin-relative metres."""
        return self._y

    @property
    def heading(self) -> float:
        """Current heading in radians. 0 = East (+X), increases counter-clockwise."""
        return self._heading

    @property
    def position(self) -> tuple[float, float]:
        """Current (x, y) position as a tuple."""
        return (self._x, self._y)

    # ── Core update ──────────────────────────────────────────────────────────

    def update(self) -> tuple[float, float, float]:
        """
        Must be called exactly once per simulation timestep, after robot.step().

        Returns:
            (x, y, heading) — same as the individual properties.
        """
        # 1. Read compass for absolute heading
        #    Webots compass returns a unit vector pointing toward North.
        #    atan2(values[0], values[1]) converts it to an angle in the XY plane.
        if self._compass:
            v = self._compass.getValues()
            self._heading = math.atan2(v[0], v[1])

        # 2. Read wheel encoders
        fl_now = self._fl.getValue() if self._fl else 0.0
        fr_now = self._fr.getValue() if self._fr else 0.0

        # 3. First call: just store baseline, don't compute delta
        if not self._initialised:
            self._prev_fl = fl_now
            self._prev_fr = fr_now
            self._initialised = True
            return (self._x, self._y, self._heading)

        # 4. Delta wheel angles (radians) → linear distance (metres)
        d_fl = (fl_now - self._prev_fl) * WHEEL_RADIUS
        d_fr = (fr_now - self._prev_fr) * WHEEL_RADIUS

        # 5. Forward distance this step (average of both wheels)
        d = (d_fl + d_fr) / 2.0

        # 6. Update position using current compass heading
        #    (not integrated heading — this is the key drift fix)
        self._x += d * math.cos(self._heading)
        self._y += d * math.sin(self._heading)

        # 7. Save encoder readings for next step
        self._prev_fl = fl_now
        self._prev_fr = fr_now

        return (self._x, self._y, self._heading)

    # ── Utility ──────────────────────────────────────────────────────────────

    def distance_to(self, target_x: float, target_y: float) -> float:
        """Euclidean distance from current position to a target point."""
        return math.hypot(target_x - self._x, target_y - self._y)

    def bearing_to(self, target_x: float, target_y: float) -> float:
        """
        Angle from current position toward the target, in radians.
        Result is in [-π, π] using the same convention as heading.
        """
        return math.atan2(target_y - self._y, target_x - self._x)

    def heading_error_to(self, target_x: float, target_y: float) -> float:
        """
        Signed heading error toward target, in radians.
        Positive = target is to the left (robot needs to turn CCW).
        Negative = target is to the right (robot needs to turn CW).
        Result is in [-π, π].
        """
        desired = self.bearing_to(target_x, target_y)
        error   = desired - self._heading
        # Normalise to [-π, π]
        while error >  math.pi: error -= 2 * math.pi
        while error < -math.pi: error += 2 * math.pi
        return error

    def __repr__(self) -> str:
        return (
            f"Odometry(x={self._x:.3f}, y={self._y:.3f}, "
            f"heading={math.degrees(self._heading):.1f}°)"
        )
