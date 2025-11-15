import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ranger_msgs.msg import SystemState, MotionState, ActuatorStateArray
from sensor_msgs.msg import BatteryState
import numpy as np
import math
import time
from simple_pid import PID
from typing import Union


class RangerMini3(Node):
    def __init__(self):
        super().__init__('agv_controller')

        # Publisher for velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(SystemState, '/system_state', self.system_error_state_callback, 10)
        # 如需启用更多状态订阅，取消下面注释：
        # self.create_subscription(SystemState, '/system_state', self.system_state_callback, 10)
        # self.create_subscription(MotionState, '/motion_state', self.motion_state_callback, 10)
        # self.create_subscription(ActuatorStateArray, '/actuator_state', self.actuator_state_callback, 10)
        # self.create_subscription(BatteryState, '/battery_state', self.battery_state_callback, 10)

        # Params
        self.linear_speed = 0.3  # m/s
        self.angular_speed = 20 * np.pi / 180  # rad/s
        self.rate_hz = 20
        self.odom_reset_flag = False

        # PID controllers
        self.linear_controller = PID(Kp=0.8, Ki=0.1, Kd=0.05, setpoint=0.0)
        self.linear_controller.output_limits = (-self.linear_speed, self.linear_speed)
        self.linear_controller.sample_time = 1.0 / self.rate_hz

        self.angular_controller = PID(Kp=0.85, Ki=0.05, Kd=0.05, setpoint=0.0)
        self.angular_controller.output_limits = (-self.angular_speed, self.angular_speed)
        self.angular_controller.sample_time = 1.0 / self.rate_hz

        self.distance_threshold = 0.01
        self.theta_threshold = np.pi / 180

        self.odom_data = {'time': [], 'distance': [], 'theta': [], 'linear_speed': [], 'angular_speed': []}

        self.current_position = None
        self.current_orientation = None
        self.start_position = None
        self.start_orientation = None

        self.vehicle_state = None
        print('ranger mini 3 is ready to move')

    # ========= Callbacks =========
    def odom_callback(self, msg: Odometry):
        self.current_position = msg.pose.pose.position
        self.current_orientation = msg.pose.pose.orientation

    def system_error_state_callback(self, state: SystemState):
        self.vehicle_state = state.vehicle_state
        # print(f"System vehicle_state: {self.vehicle_state}")

    def system_state_callback(self, state: SystemState):
        self.system_state = state
        print(f"System State: {state}")

    def motion_state_callback(self, state: MotionState):
        self.motion_state = state
        print(f"Motion State: {state}")

    def actuator_state_callback(self, state: ActuatorStateArray):
        self.actuator_state = state
        print(f"Actuator State Array: {state}")

    def battery_state_callback(self, state: BatteryState):
        self.battery_state = state
        print(f"Battery State: {state}")

    # ========= Helpers =========
    def normalize_angle(self, theta: float) -> float:
        while theta > np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        return theta

    def quaternion_to_euler(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z

    def now_sec(self) -> float:
        # ROS2 clock time (steady, simulated if available)
        nsec = self.get_clock().now().nanoseconds
        return nsec / 1e9

    # ========= Core functions =========
    def get_distance_moved(self) -> float:
        assert self.start_position is not None and self.current_position is not None
        dx = self.current_position.x - self.start_position.x
        dy = self.current_position.y - self.start_position.y
        return math.sqrt(dx ** 2 + dy ** 2)

    def get_current_yaw(self) -> float:
        assert self.start_orientation is not None and self.current_orientation is not None
        sqx, sqy, sqz, sqw = (
            self.start_orientation.x, self.start_orientation.y,
            self.start_orientation.z, self.start_orientation.w
        )
        cx, cy, cz, cw = (
            self.current_orientation.x, self.current_orientation.y,
            self.current_orientation.z, self.current_orientation.w
        )
        _, _, start_yaw = self.quaternion_to_euler(sqx, sqy, sqz, sqw)
        _, _, current_yaw = self.quaternion_to_euler(cx, cy, cz, cw)
        yaw_diff = self.normalize_angle(current_yaw - start_yaw)
        return yaw_diff

    def agv_move(self, linear_speed: float = None, angular_speed: float = None, duration: float = 0.0):
        linear_speed = float(linear_speed) if linear_speed is not None else 0.0
        angular_speed = float(angular_speed) if angular_speed is not None else 0.0
        end_time = self.now_sec() + duration
        while self.now_sec() < end_time:
            twist = Twist()
            twist.linear.x = linear_speed
            twist.angular.z = angular_speed
            self.cmd_pub.publish(twist)
            # 处理回调，保证里程计在动
            rclpy.spin_once(self, timeout_sec=duration)

    def linear_or_angular_control_loop(self, distance: float = 0.0, theta: float = 0.0):
        if distance == 0.0 and theta == 0.0:
            print("both distance and theta are 0, no need to move.")
            return

        if distance != 0.0:
            assert theta == 0.0, "when move straight, theta should be zero."
            print(f"Moving straight for distance: {distance:.3f} m")
        else:
            assert distance == 0.0, "when rotation, distance should be zero."
            theta = self.normalize_angle(theta)
            print(f"Rotating to relative theta: {theta:.3f} rad")

        # set PID setpoints
        self.linear_controller.setpoint = distance
        self.angular_controller.setpoint = theta

        # wait for first odom if not ready
        t0 = self.now_sec()
        self.start_position = self.current_position
        self.start_orientation = self.current_orientation
        while self.start_position is None or self.start_orientation is None:
            self.start_position = self.current_position
            self.start_orientation = self.current_orientation
            rclpy.spin_once(self, timeout_sec=0.1)

        loop_dt = 1.0 / self.rate_hz
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)

            if self.vehicle_state == 2:  # Emergency Stop
                print("Emergency stop detected, aborting motion.")
                return

            current_distance = self.get_distance_moved()
            if distance < 0:
                current_distance = -current_distance
            current_theta = self.get_current_yaw()

            if (abs(distance - current_distance) < self.distance_threshold and
                    abs(theta - current_theta) < self.theta_threshold):
                self.linear_controller.reset()
                self.angular_controller.reset()
                rclpy.spin_once(self, timeout_sec=2.0)
                break
            else:
                lin = self.linear_controller(current_distance)
                ang = self.angular_controller(current_theta)
                print(
                    f"target_d={distance:.3f}, d={current_distance:.3f}, "
                    f"target_th={theta:.3f}, th={current_theta:.3f}, "
                    f"cmd=({lin:.3f}, {ang:.3f})"
                )
                self.agv_move(linear_speed=lin, angular_speed=ang, duration=loop_dt)

    def motion(self, paths):
        points = np.array(paths, dtype=float)
        ts = points[:, 2]
        thetas = np.zeros_like(ts)
        thetas[0] = self.normalize_angle(ts[0])
        for i in range(1, len(thetas)):
            thetas[i] = self.normalize_angle(ts[i] - ts[i - 1])

        distances = np.linalg.norm(points[1:, :2] - points[:-1, :2], axis=1)
        assert len(thetas) == len(distances) + 1

        ways = np.zeros(len(thetas) + len(distances))
        ways[0::2] = thetas
        ways[1::2] = distances

        for idx, w in enumerate(ways):
            print(idx, w)
            if idx % 2 == 0:
                self.linear_or_angular_control_loop(theta=w)
            else:
                self.linear_or_angular_control_loop(distance=w)

    def terminate(self):
        self.cmd_pub.publish(Twist())  # stop
        print('Terminated and stopped the robot.')


def main():
    rclpy.init()
    node = RangerMini3()
    try:
        paths = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, np.pi],
            [0.0, 0.0, np.pi],
        ]
        node.motion(paths)
        # 示例：
        # node.linear_or_angular_control_loop(distance=2.0)
        # node.linear_or_angular_control_loop(theta=2*np.pi)
    except KeyboardInterrupt:
        pass
    finally:
        node.terminate()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
