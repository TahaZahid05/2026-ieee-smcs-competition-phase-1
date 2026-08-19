"""
Basic Rosbot Controller for Search and Rescue Simulation
Implements cooperative multi-target exploration with live occupancy mapping and A* replanning.
"""
from controller import Robot
import json
import math
import os
from odometry import Odometry
from path_planner import PathPlanner
from navigator import Navigator
from live_map import LiveMap

class BasicRosbotController:
    def __init__(self):
        """Initialize the robot controller"""
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Get robot name for identification
        self.robot_id = self.robot.getName()

        # Initialize sensors and actuators
        self._init_devices()

        # Navigation constants
        self.max_speed = 5.0

        # Load known victim candidate locations
        self.all_victims = self._load_victims()
        self.visited_victims = set()
        self.claimed_by_peer = {}

        if self.robot_id == "robot1":
            start_x, start_y = -0.375, 0.375
            initial_target = (2.12, 5.71)   # Victim 3 (Dining Room)
        else:
            start_x, start_y = -0.375, 0.0
            initial_target = (3.99, -3.94)  # Victim 2 (Bedroom open floor)

        self._target_x, self._target_y = initial_target
        self._current_target = initial_target
        self._broadcast_pursuing(initial_target)

        self.odom = Odometry(
            compass=self.compass,
            fl_sensor=self.front_left_position_sensor,
            fr_sensor=self.front_right_position_sensor,
            initial_x=start_x,
            initial_y=start_y,
        )

        self.planner = PathPlanner(
            "sim_logs/map_estimate.png",
            "sim_logs/map_metadata.json"
        )

        self.live_map = LiveMap(
            "sim_logs/map_estimate.png",
            "sim_logs/map_metadata.json"
        )
        self._last_replan_time = 0.0
        self._replan_interval  = 3.0   # seconds between A* replans

        self.waypoints = self.planner.plan(
            self.odom.x,
            self.odom.y,
            self._target_x,
            self._target_y,
        )

        self.navigator = Navigator(
            robot_id=self.robot_id,
            odometry=self.odom,
            set_speeds_fn=self.set_wheel_speeds
        )

        if self.waypoints:
            self.navigator.set_waypoints(self.waypoints)

        print(f"[{self.robot_id}] Initialized - Target: ({self._target_x:.2f}, {self._target_y:.2f})")

    def _load_victims(self) -> list[tuple[float, float]]:
        """Load estimated victim coordinates from CSV or defaults."""
        victims = []
        csv_path = "sim_logs/victim_location_estimates.csv"
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(",")
                        victims.append((float(parts[0]), float(parts[1])))
        if not victims:
            victims = [
                (2.71, -3.22),   # victim 1 (bedroom bed)
                (3.99, -3.94),   # victim 2 (bedroom open)
                (2.12, 5.71),    # victim 3 (dining table)
                (8.67, 6.54),    # victim 4 (northeast room)
            ]
        return victims

    def _is_victim_visited(self, target_coord: tuple[float, float]) -> bool:
        """Check if target coordinate is close to an already visited victim."""
        for v in self.visited_victims:
            if math.hypot(target_coord[0] - v[0], target_coord[1] - v[1]) < 0.8:
                return True
        return False

    def _mark_victim_visited(self, coord: tuple[float, float]):
        """Mark a victim coordinate as visited."""
        for v in self.all_victims:
            if math.hypot(coord[0] - v[0], coord[1] - v[1]) < 0.8:
                self.visited_victims.add(v)

    def _get_next_target(self) -> tuple[float, float] | None:
        """Find the closest unvisited and unclaimed victim to the robot's current position."""
        best_target = None
        min_dist = float("inf")
        claimed_coords = list(self.claimed_by_peer.values())

        for vx, vy in self.all_victims:
            if self._is_victim_visited((vx, vy)):
                continue
            # Skip if peer is actively pursuing this victim
            if any(math.hypot(vx - cx, vy - cy) < 0.8 for cx, cy in claimed_coords):
                continue
            d = self.odom.distance_to(vx, vy)
            if d < min_dist:
                min_dist = d
                best_target = (vx, vy)

        # Fallback: if all remaining unvisited are claimed, pick closest unvisited
        if best_target is None:
            for vx, vy in self.all_victims:
                if not self._is_victim_visited((vx, vy)):
                    d = self.odom.distance_to(vx, vy)
                    if d < min_dist:
                        min_dist = d
                        best_target = (vx, vy)

        return best_target

    def _assign_next_target(self) -> bool:
        """Select, broadcast, and route to the next available victim."""
        next_target = self._get_next_target()
        if next_target is not None:
            self._target_x, self._target_y = next_target
            self._current_target = next_target
            self._broadcast_pursuing(next_target)
            live_grid = self.live_map.get_inflated_grid(inflation_px=5)
            wps = self.planner.replan_on_grid(
                live_grid,
                self.odom.x,
                self.odom.y,
                self._target_x,
                self._target_y,
            )
            if wps:
                self.navigator.set_waypoints(wps)
                print(f"[{self.robot_id}] Next target assigned: ({self._target_x:.2f}, {self._target_y:.2f}), {len(wps)} waypoints")
            else:
                print(f"[{self.robot_id}] Warning: Could not plan path to ({self._target_x:.2f}, {self._target_y:.2f})")
            self._last_replan_time = self.robot.getTime()
            return True
        else:
            print(f"[{self.robot_id}] All victims found or assigned! Mission complete.")
            self._current_target = None
            self.navigator.set_waypoints([])
            return False

    def _broadcast_pursuing(self, coord: tuple[float, float]):
        """Inform teammate robot that we are currently pursuing target at coord."""
        if self.squad_emitter:
            msg = json.dumps({
                "type": "pursuing_target",
                "robot_id": self.robot_id,
                "coord": list(coord)
            })
            self.squad_emitter.send(msg.encode())

    def _broadcast_victim_found(self, coord: tuple[float, float]):
        """Inform teammate robot that a victim was found at coord."""
        if self.squad_emitter:
            msg = json.dumps({
                "type": "victim_found",
                "robot_id": self.robot_id,
                "coord": list(coord)
            })
            self.squad_emitter.send(msg.encode())

    def _check_squad_messages(self):
        """Receive updates from teammate robot."""
        if not self.squad_receiver:
            return
        while self.squad_receiver.getQueueLength() > 0:
            data_str = self.squad_receiver.getString()
            self.squad_receiver.nextPacket()
            try:
                msg = json.loads(data_str)
                sender_id = msg.get("robot_id")
                msg_type = msg.get("type")
                coord = tuple(msg.get("coord"))

                if msg_type == "pursuing_target":
                    self.claimed_by_peer[sender_id] = coord
                    # If we are both pursuing the same target, re-evaluate
                    if self._current_target is not None:
                        if math.hypot(self._current_target[0] - coord[0], self._current_target[1] - coord[1]) < 0.8:
                            # Higher robot ID or further distance yields to peer
                            peer_dist = math.hypot(coord[0], coord[1])
                            my_dist = self.odom.distance_to(self._current_target[0], self._current_target[1])
                            if self.robot_id > sender_id:
                                print(f"[{self.robot_id}] Peer {sender_id} is also pursuing {coord}. Re-routing...")
                                self._assign_next_target()

                elif msg_type == "victim_found":
                    self._mark_victim_visited(coord)
                    if sender_id in self.claimed_by_peer:
                        del self.claimed_by_peer[sender_id]
                    # If we were targeting this same victim, re-route
                    if self._current_target is not None:
                        if math.hypot(self._current_target[0] - coord[0], self._current_target[1] - coord[1]) < 0.8:
                            print(f"[{self.robot_id}] Current target {coord} was found by peer! Re-routing...")
                            self._assign_next_target()
            except Exception as e:
                pass

    def _init_devices(self):
        """Initialize robot sensors and actuators"""
        # Motors
        self.front_left_motor = self.robot.getDevice("fl_wheel_joint")
        self.front_right_motor = self.robot.getDevice("fr_wheel_joint")
        self.rear_left_motor = self.robot.getDevice("rl_wheel_joint")
        self.rear_right_motor = self.robot.getDevice("rr_wheel_joint")
        self.front_left_motor.setPosition(float("inf"))
        self.front_right_motor.setPosition(float("inf"))
        self.rear_left_motor.setPosition(float("inf"))
        self.rear_right_motor.setPosition(float("inf"))
        self.front_left_motor.setVelocity(0)
        self.front_right_motor.setVelocity(0)
        self.rear_left_motor.setVelocity(0)
        self.rear_right_motor.setVelocity(0)

        # Wheel position sensors
        self.front_left_position_sensor = self.robot.getDevice(
            "front left wheel motor sensor"
        )
        self.front_right_position_sensor = self.robot.getDevice(
            "front right wheel motor sensor"
        )
        self.rear_left_position_sensor = self.robot.getDevice(
            "rear left wheel motor sensor"
        )
        self.rear_right_position_sensor = self.robot.getDevice(
            "rear right wheel motor sensor"
        )
        self.front_left_position_sensor.enable(self.timestep)
        self.front_right_position_sensor.enable(self.timestep)
        self.rear_left_position_sensor.enable(self.timestep)
        self.rear_right_position_sensor.enable(self.timestep)

        # RGB camera
        try:
            self.camera_rgb = self.robot.getDevice("camera rgb")
            self.camera_rgb.enable(self.timestep)
        except:
            self.camera_rgb = None
            print(f"[{self.robot_id}] Warning: No RGB camera found")

        # Depth camera
        try:
            self.camera_depth = self.robot.getDevice("camera depth")
            self.camera_depth.enable(self.timestep)
        except:
            self.camera_depth = None
            print(f"[{self.robot_id}] Warning: No depth camera found")

        # Lidar sensor
        try:
            self.lidar = self.robot.getDevice("laser")
            self.lidar.enable(self.timestep)
        except:
            self.lidar = None
            print(f"[{self.robot_id}] Warning: No lidar found")

        # Accelerometer
        try:
            self.accelerometer = self.robot.getDevice("imu accelerometer")
            self.accelerometer.enable(self.timestep)
        except:
            print(f"[{self.robot_id}] Warning: No accelerometer found")
            self.accelerometer = None

        # Gyro
        try:
            self.gyro = self.robot.getDevice("imu gyro")
            self.gyro.enable(self.timestep)
        except:
            print(f"[{self.robot_id}] Warning: No gyro found")
            self.gyro = None

        # Compass
        try:
            self.compass = self.robot.getDevice("imu compass")
            self.compass.enable(self.timestep)
        except:
            print(f"[{self.robot_id}] Warning: No compass found")
            self.compass = None

        # Distance sensors
        self.distance_sensors = []
        sensor_names = ["fl_range", "fr_range", "rl_range", "rr_range"]
        for name in sensor_names:
            try:
                sensor = self.robot.getDevice(name)
                sensor.enable(self.timestep)
                self.distance_sensors.append(sensor)
            except:
                print(f"[{self.robot_id}] Warning: No {name} sensor found")

        # Communication devices to supervisor
        try:
            self.supervisor_emitter = self.robot.getDevice("supervisor emitter")
        except:
            print(
                f"[{self.robot_id}] Warning: Supervisor communication devices not found"
            )
            self.supervisor_emitter = None

        # Communication devices for robot to robot communication
        try:
            self.squad_receiver = self.robot.getDevice("robot to robot receiver")
            self.squad_receiver.enable(self.timestep)

            self.squad_emitter = self.robot.getDevice("robot to robot emitter")
        except:
            print(
                f"[{self.robot_id}] Warning: Robot to robot communication devices not found"
            )
            self.squad_receiver = None
            self.squad_emitter = None

    def send_victim_found_message(
        self,
        victim_detected: bool = False,
        confidence: float = 0.0,
    ):
        """Send victim found message to marking supervisor"""
        if not self.supervisor_emitter:
            print(f"[{self.robot_id}] Warning: Cannot send request - no emitter")
            return

        request = {
            "timestamp": self.robot.getTime(),
            "robot_id": self.robot_id,
            "position": [self.odom.x, self.odom.y, 0.0],
            "victim_found": victim_detected,
            "victim_confidence": confidence,
        }

        message = json.dumps(request)
        self.supervisor_emitter.send(message.encode())

        if victim_detected:
            print(f"[{self.robot_id}] VICTIM ALERT: Confidence {confidence:.1%}")

    def set_wheel_speeds(self, left_speed: float, right_speed: float):
        """Set wheel motor speeds"""
        left_speed = max(-self.max_speed, min(self.max_speed, left_speed))
        right_speed = max(-self.max_speed, min(self.max_speed, right_speed))

        self.front_left_motor.setVelocity(left_speed)
        self.rear_left_motor.setVelocity(left_speed)
        self.front_right_motor.setVelocity(right_speed)
        self.rear_right_motor.setVelocity(right_speed)

    def set_explore_behavior(self):
        """Navigate to victims using live occupancy grid mapping and A* replanning."""
        current_time = self.robot.getTime()

        # 1. Process messages from squad teammate
        self._check_squad_messages()

        # 2. Update live map with current Lidar hits
        self.live_map.update(self.odom.x, self.odom.y, self.odom.heading, self.lidar)

        # 3. Handle arrival at current victim target
        if self.navigator.is_arrived:
            if self._current_target is not None:
                dist_to_target = self.odom.distance_to(self._target_x, self._target_y)
                if dist_to_target < 0.6:
                    print(f"[{self.robot_id}] Reached target ({self._target_x:.2f}, {self._target_y:.2f}) (dist={dist_to_target:.2f}m)! Reporting to supervisor...")
                    self.send_victim_found_message(True, 1.0)
                    self._broadcast_victim_found(self._current_target)
                    self._mark_victim_visited(self._current_target)
                    self._assign_next_target()
            return

        # 4. Periodic A* replan on the live map (every 3 seconds)
        if (
            self._current_target is not None
            and current_time - self._last_replan_time >= self._replan_interval
        ):
            live_grid = self.live_map.get_inflated_grid(inflation_px=5)
            new_wps = self.planner.replan_on_grid(
                live_grid,
                self.odom.x,
                self.odom.y,
                self._target_x,
                self._target_y,
            )
            if new_wps:
                self.navigator.set_waypoints(new_wps)
            self._last_replan_time = current_time

        # 5. Pure waypoint tracking
        status = self.navigator.step()

        # 6. Periodic position + status log every 5 seconds
        if not hasattr(self, '_last_log_time'):
            self._last_log_time = 0.0
        if current_time - self._last_log_time >= 5.0:
            wp = self.navigator.current_target
            wp_str = f"({wp[0]:.2f},{wp[1]:.2f})" if wp else "none"
            tgt_str = f"({self._target_x:.2f},{self._target_y:.2f})" if self._current_target else "none"
            print(f"[{self.robot_id}] t={current_time:.0f}s "
                  f"pos=({self.odom.x:.2f},{self.odom.y:.2f}) "
                  f"hdg={math.degrees(self.odom.heading):.0f}° "
                  f"mode={status} target={tgt_str} next_wp={wp_str}")
            self._last_log_time = current_time

    def run(self):
        """Main robot control loop"""
        print(f"[{self.robot_id}] Starting search and rescue mission")

        while self.robot.step(self.timestep) != -1:
            self.odom.update()

            # Stagger start: robot1 waits 8s for robot2 to clear the shared corridor
            if self.robot_id == "robot1" and self.robot.getTime() < 8.0:
                continue

            self.set_explore_behavior()


def main():
    """Main function to run the robot controller"""
    controller = BasicRosbotController()
    controller.run()


if __name__ == "__main__":
    main()
