import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ranger_msgs.msg import SystemState, MotionState, ActuatorStateArray, BatteryState
import numpy as np
import math
import time
from simple_pid import PID
# from scipy.interpolate import interp1d
from typing import Union


"""
# at ROS
# it is easy using in ros and ros2
# run this scrip, you should running 
source ~/agilex_ws/devel/setup.bash
rosrun ranger_bringup bringup_can2usb.bash
roslaunch ranger_bringup ranger_mini_v2.launch
"""

class RangerMini3:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('agv_controller', anonymous=True)
        
        # Publisher for velocity commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Subscriber for odometry
        rospy.Subscriber('/ranger_base_node/odom', Odometry, self.odom_callback)
        # Other Subscriber for ranger mini 3
        rospy.Subscriber('/system_state', SystemState, self.system_error_state_callback)
        # rospy.Subscriber("/system_state", SystemState, self.system_state_callback)
        # rospy.Subscriber("/motion_state", MotionState, self.motion_state_callback)
        # rospy.Subscriber("/actuator_state", ActuatorStateArray, self.actuator_state_callback)
        # rospy.Subscriber("/battery_state", BatteryState, self.battery_state_callback)


        self.linear_speed = 0.3  # m/s
        self.angular_speed = 20 * np.pi / 180  # rad/s
        self.rate_hz = 20
        self.odom_reset_flag = False

        # Initialize PID controllers
        # self.linear_controller = PID(Kp=0.6, Ki=0.1, Kd=0.0, setpoint=0.0)
        # self.linear_controller.output_limits = (-self.linear_speed, self.linear_speed)
        # self.linear_controller.sample_time = 1.0 / self.rate_hz

        # self.angular_controller = PID(Kp=0.6, Ki=0.1, Kd=0.0, setpoint=0.0)
        # self.angular_controller.output_limits = (-self.angular_speed, self.angular_speed)
        # self.angular_controller.sample_time = 1.0 / self.rate_hz

        # Initialize PID controllers with adjusted parameters

        # self.linear_controller = PID(Kp=0.3, Ki=0.1, Kd=0.1, setpoint=0.0)
        self.linear_controller = PID(Kp=0.8, Ki=0.1, Kd=0.05, setpoint=0.0)

        self.linear_controller.output_limits = (-self.linear_speed, self.linear_speed)
        self.linear_controller.sample_time = 1.0 / self.rate_hz  # Example: rate_hz = 10 for 10 Hz update rate

        self.angular_controller = PID(Kp=0.85, Ki=0.05, Kd=0.05, setpoint=0.0)
        self.angular_controller.output_limits = (-self.angular_speed, self.angular_speed)
        self.angular_controller.sample_time = 1.0 / self.rate_hz


        self.distance_threshold = 0.01
        self.thete_threshold = np.pi / 180

        self.odom_data = {
            'time': [],
            'distance': [],
            'theta': [],
            'linear_speed': [],
            'angular_speed': []
        }

        self.current_position = None
        self.current_orientation = None
        self.start_position = None
        self.start_orientation = None

        self.vehicle_state = None
        print("ranger mini 3 is ready to move")

    # def normalize_angle(self, theta):
    #     theta = (theta + np.pi) % (2 * np.pi) - np.pi

    def normalize_angle(self, theta):
        while theta > np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        return theta
    
    def odom_callback(self, msg):
        """Handle the odometry data and update the robot's current position and orientation."""
        self.current_position = msg.pose.pose.position
        self.current_orientation = msg.pose.pose.orientation

    def system_error_state_callback(self, state):
        self.vehicle_state = state.vehicle_state
        # control_mode = state.control_mode
        # error_code = state.error_code
        # rospy.loginfo("System State: %s", self.vehicle_state)

    def system_state_callback(self, state):
        self.system_state = state
        rospy.loginfo("System State: %s", state)

    def motion_state_callback(self, state):
        self.motion_state = state
        rospy.loginfo("Motion State: %s", state)

    def actuator_state_callback(self, state):
        self.actuator_state = state
        rospy.loginfo("Actuator State Array: %s", state)

    def battery_state_callback(self, state):
        self.battery_state = state
        rospy.loginfo("Battery State: %s", state)

    def get_distance_moved(self):
        """Calculates the distance moved by the robot based on the start and current position."""
        assert self.start_position and self.current_position
        dx = self.current_position.x - self.start_position.x
        dy = self.current_position.y - self.start_position.y
        distance_moved = math.sqrt(dx ** 2 + dy ** 2)
        return distance_moved

    def get_current_yaw(self):
        """Calculates the current yaw (rotation around z-axis) based on the robot's orientation."""
        assert self.start_orientation and self.current_orientation
        # Yaw is the rotation around z-axis
        start_quaternion = (
            self.start_orientation.x,
            self.start_orientation.y,
            self.start_orientation.z,
            self.start_orientation.w
        )
        current_quaternion = (
            self.current_orientation.x,
            self.current_orientation.y,
            self.current_orientation.z,
            self.current_orientation.w
        )

        # Convert quaternions to euler angles
        start_euler = self.quaternion_to_euler(*start_quaternion)
        current_euler = self.quaternion_to_euler(*current_quaternion)

        # Extract yaw (z-axis rotation)
        start_yaw = start_euler[2]
        current_yaw = current_euler[2]

        # Calculate yaw difference
        yaw_difference = current_yaw - start_yaw
        yaw_difference = self.normalize_angle(yaw_difference)
        # assert yaw_difference < np.pi
        return yaw_difference

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to euler angles (roll, pitch, yaw)."""
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

    def agv_move(self, linear_speed: float = None, angular_speed: float = None, duration: float = 0.0):
        """Move the robot with specified linear and angular speeds for a duration."""
        linear_speed = float(linear_speed) if linear_speed is not None else 0.0
        angular_speed = float(angular_speed) if angular_speed is not None else 0.0
        end_time = rospy.Time.now().to_sec() + duration
        while rospy.Time.now().to_sec() < end_time:
            twist_message = Twist()
            twist_message.linear.x = linear_speed
            twist_message.angular.z = angular_speed
            self.cmd_vel_pub.publish(twist_message)
            rospy.sleep(duration)  # Sleep to simulate control loop timing

    def linear_or_angular_control_loop(self, distance: float=0.0, theta: float=0.0):
        """Control the robot's movement to move forward a specified distance."""
        if distance == 0.0 and theta == 0.0:
            print("both distance and theta is 0, no need to move.")
            return

        # theta = 0  # Keep the robot straight
        # distance = 0  # Keep the robot rotation
        if distance != 0:
            assert theta == 0, "when move straight, theta should be zero."
            print(f"Moving straight for distance: {distance}")
        else:
            assert distance == 0, "when rotation, distance should be zero."
            print(f"Rotating to relative theta: {theta}")
            theta = self.normalize_angle(theta=theta)

        self.linear_controller.setpoint = distance
        self.angular_controller.setpoint = theta

        # init start position and orientation
        self.start_position = self.current_position
        self.start_orientation = self.current_orientation

        while self.start_position is None or self.start_orientation is None:
            self.start_position = self.current_position
            self.start_orientation = self.current_orientation
            rospy.sleep(0.1)

        while True:
            if self.vehicle_state == 2:  # Emergency Stop
                return
            current_distance = self.get_distance_moved()
            if distance < 0:
                current_distance = -current_distance
            current_theta = self.get_current_yaw()

            if abs(distance - current_distance) < self.distance_threshold and \
                    abs(theta - current_theta) < self.thete_threshold:
                self.linear_controller.reset()
                self.angular_controller.reset()
                rospy.sleep(2)
                break
            else:
                linear_speed = self.linear_controller(current_distance)
                angular_speed = self.angular_controller(current_theta)

                print(distance, current_distance, abs(distance - current_distance), linear_speed, theta, current_theta, abs(theta - current_theta), angular_speed)
                self.agv_move(linear_speed=linear_speed, angular_speed=angular_speed, duration=1.0 / self.rate_hz)


    def motion(self, paths):
        points = np.array(paths)
        ts = points[:, 2]
        thetas = np.zeros_like(ts)
        thetas[0] = self.normalize_angle(ts[0])
        for i in range(1, len(thetas)):
            thetas[i] = self.normalize_angle(theta=ts[i] - ts[i - 1])

        # just can be positive, if you want to use nagitave, you can use linear_or_angular_control_loop straightly.
        distances = np.linalg.norm(points[1:, :2] - points[:-1, :2], axis=1)
        assert len(thetas) == len(distances) + 1
        ways = np.zeros(len(thetas) + len(distances))
        ways[::2] = thetas

        ways[1::2] = distances
        for cnt in range(len(ways)):
            if cnt % 2 == 0:
                self.linear_or_angular_control_loop(theta=ways[cnt])
            else:
                self.linear_or_angular_control_loop(distance=ways[cnt])

    def terminate(self):
        """Stop all movement and cleanup."""
        self.cmd_vel_pub.publish(Twist())  # Send zero velocities to stop the robot
        print('Terminated and stopped the robot.')


if __name__ == '__main__':
    robot_controller = RangerMini3()

    try:
        paths = [
            [0.0,        0.0,         0],
            [1.0,          0.0,         np.pi],
            [0.0,        0.0,         np.pi],
        ]
        robot_controller.motion(paths)
        # robot_controller.linear_or_angular_control_loop(distance=2)
        # robot_controller.linear_or_angular_control_loop(theta=2 * np.pi)
        # robot_controller.linear_or_angular_control_loop(theta=-2 * np.pi)
        # robot_controller.linear_or_angular_control_loop(theta=2 * np.pi)
        # robot_controller.linear_or_angular_control_loop(theta=-math.pi)
        # robot_controller.linear_or_angular_control_loop(theta=math.pi)
        # robot_controller.linear_or_angular_control_loop(theta=-math.pi)
        # robot_controller.linear_or_angular_control_loop(distance=-1)


    except rospy.ROSInterruptException:
        pass
    finally:
        robot_controller.terminate()
