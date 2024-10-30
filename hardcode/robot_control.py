from arm.xarm6 import XARM6
from base.ranger_mini_3 import RangerMini3
from camera.realsense_d455 import RealSense_D455
from transforms3d.quaternions import mat2quat, quat2mat
from utils.utils import quat2rpy, rpy_to_rotation_matrix, get_pose_look_at
from utils.utils import wrist_serial_number, top_serial_number, calib_result_path
import numpy as np
import open3d as o3d
import pickle
import time
import copy

"""
# run this scrip, you should running
source ~/agilex_ws/devel/setup.bash
rosrun ranger_bringup bringup_can2usb.bash

# on the other script
roslaunch ranger_bringup ranger_mini_v2.launch
"""

class RobotControl:
    def __init__(
        self,
        gripper_length=0.172,
        servo_speed=20
    ):
        try:
            self.arm = XARM6()
            self.base = RangerMini3()
            self.camera_wrist = RealSense_D455(WH=[1280, 720], depth_threshold=[0, 2], serial_number=wrist_serial_number, remove_robot=True)
            # self.camera_top = RealSense_D455(WH=[1280, 720], depth_threshold=[0.35, 2], serial_number=top_serial_number, remove_robot=True, FPS=5)
        except Exception as e:
            print(e)
            self.arm.reset()
            self.base.terminate()
            self.camera_wrist.pipeline.close()
        
        self.gripper_length = gripper_length
        self.servo_speed = servo_speed

        # Initialize the calibration of the wrist camera
        self._init_calibration()

        self.camera_top_pose = np.eye(4)
        self.camera_top_pose[:3, 3] = [-0.31, 0, 0.94]

        self.camera_wrist_positions = {
            "camera_0": np.array([0.3, 0, 0.65]),
            "camera_1": np.array([0.35, 0.2, 0.45]),
            "camera_2": np.array([0.3, 0, 0.65]),
            "camera_3": np.array([0.35, -0.2, 0.45]),
        }

        target_far = np.array([0.8, 0, 0.25])
        target_low = np.array([0.8, 0, 0.1])

        self.camera_wrist_poses = [
            get_pose_look_at(
                eye=np.array(self.camera_wrist_positions["camera_0"]), target=target_far
            ),
            get_pose_look_at(
                eye=np.array(self.camera_wrist_positions["camera_1"]), target=target_far
            ),
            get_pose_look_at(
                eye=np.array(self.camera_wrist_positions["camera_2"]), target=target_low
            ),
            get_pose_look_at(
                eye=np.array(self.camera_wrist_positions["camera_3"]), target=target_far
            ),
        ]

        self.see_poses = [
            [0, 0, -180, 0, 135, 0],
            [0, 0, -180, 0, 145, 0],
            [0, 0, -180, 0, 155, 0],
            [15, 0, -180, 0, 155, 0],
            [15, 0, -180, 0, 145, 0],
            [15, 0, -180, 0, 135, 0],
            [-15, 0, -180, 0, 135, 0],
            [-15, 0, -180, 0, 145, 0],
            [-15, 0, -180, 0, 155, 0],
        ]

        self.see_back_pose = [180, 0, -180, 0, 150, 0]
        self.place_back_pose = [180, 0, -110, 0, 110, 0]

        # self.see_poses = [
        #     [0, 0, 0, 0, -90, 0],
        #     [15, 0, 0, 0, -90, 0],
        #     [-15, 0, 0, 0, -90, 0],
        # ]

    def _init_calibration(self):
        self.cam2gripper = np.eye(4)
        self.cam2gripper[:3, :3] = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        # self.cam2gripper[:3, 3] = np.array([ 0.07127579,  0.01343263, -0.1270392])
        # self.cam2gripper[:3, 3] = np.array([0.0675, 0.0115, -0.14814])
        # self.cam2gripper[:3, 3] = np.array([0.0675, 0.0115, -0.14864])
        self.cam2gripper[:3, 3] = np.array([0.0675, 0.0115, -0.14864])
        # self.cam2gripper[:3, 3] = np.array([0.0675, 0.0135, -0.148])
        # self.cam2gripper[:3, 3] = np.array([0.07, 0.0115, -0.14814])

    def _get_cam2base(self):
        current_pose = self.arm.get_current_pose()
        print("Current pose: ", current_pose)
        R_cur_gripper2base = rpy_to_rotation_matrix(
            current_pose[3], current_pose[4], current_pose[5]
        )
        # Need to make srue the unit of the pose is in meters
        t_cur_gripper2base = np.array(current_pose[:3]) / 1000
        current_gripper2base = np.eye(4)
        current_gripper2base[:3, :3] = R_cur_gripper2base
        current_gripper2base[:3, 3] = t_cur_gripper2base

        # Get the camera pose in the base frame
        cam2base = np.dot(current_gripper2base, self.cam2gripper)

        # return current_gripper2base
        return cam2base


    def get_observations(self, just_wrist=False, **kwargs):
        observations = {}
        observations = {"top": None, "wrist": {}}
        if not just_wrist:
            # Get the observations from top camera
            points, colors, depths, mask = self.camera_top.get_observations()
            observations["top"] = {
                "point": points,
                "rgb": colors,
                "depth": depths,
                "mask": mask,
                "c2b": self.camera_top_pose,
                "intrinsic": self.camera_top.intrinsic_matrix,
                "dist_coef": self.camera_top.dist_coef,
            }
        # Get the observations from wrist camera
        for i, pose in enumerate(self.see_poses):
            self.arm._arm.set_servo_angle(
                angle=pose, speed=self.servo_speed, is_radian=False, wait=True
            )

            time.sleep(1)
            points, colors, depths, mask = self.camera_wrist.get_observations()

            observations["wrist"][i] = {
                "point": points,
                "rgb": colors,
                "depth": depths,
                "mask": mask,
                "c2b": self._get_cam2base(),
                "intrinsic": self.camera_wrist.intrinsic_matrix,
                "dist_coef": self.camera_wrist.dist_coef,
            }

        if False:
            pcds = []
            for name, obs in observations["wrist"].items():
                # Get the rgb, point cloud, and the camera pose
                color = obs["rgb"]
                point = obs["point"]
                mask = obs["mask"]
                c2b = obs["c2b"]
                # c2b_new = copy.deepcopy(c2b)
                # point_new = point @ c2b_new[:3, :3].T + c2b_new[:3, 3]
                point_new = point @ c2b[:3, :3].T + c2b[:3, 3]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_new[mask])
                pcd.colors = o3d.utility.Vector3dVector(color[mask])
                pcds.append(pcd)
            o3d.visualization.draw_geometries(pcds)
            
        return observations
    
    def arm_run_action(
        self, action_code=0, action_parameters=[], for_camera=False, speed=50, **kwargs
    ):
        print(f"Running action {action_code} with parameters {action_parameters}")
        if action_code == 1:
            # move the end effector, parameters: qpos

            quat = mat2quat(action_parameters[:3, :3])
            xyz = action_parameters[:3, 3] * 1000
            rpy = quat2rpy(quat)
            pose = list(xyz) + list(rpy)
            self.arm.move_to_pose(pose, wait=True, ignore_error=True, speed=speed)

            # # Need to do a 30-degree rotation in pitch if the movement is for the camera
            # # if for_camera:
            # #     rpy[1] += 30
            # self.arm.move_to_pose(list(xyz) + list(rpy), speed=speed)
            # self.arm.move_to_pose(pose=action_parameters, wait=True, ignore_error=True, speed=speed)
            # if for_camera:
            #     time.sleep(1)
        elif action_code == 2:
            # Open the gripper
            half_open = False
            if len(action_parameters) > 0:
                half_open = True
            self.arm.open_gripper(half_open=half_open)
        elif action_code == 3:
            # Close the gripper
            self.arm.close_gripper()
        elif action_code == 4:
            # Make the arm back to the default position
            self.arm.reset()

    def get_back_observation(self):
        self.arm._arm.set_servo_angle(
            angle=self.see_back_pose, 
            speed=self.servo_speed, 
            is_radian=False, 
            wait=True
        )
        time.sleep(1)
        back_observation = {}
        points, colors, depths, mask = self.camera_wrist.get_observations()
        back_observation[0] = {
            "point": points,
            "rgb": colors,
            "depth": depths,
            "mask": mask,
            "c2b": self._get_cam2base(),
            "intrinsic": self.camera_wrist.intrinsic_matrix,
            "dist_coef": self.camera_wrist.dist_coef,
        }
        return back_observation


    def pick_up(
        self, action_parameters=[], speed=50, pick_back=False
    ):
        print(f"Running action: pick_up with parameters {action_parameters} {speed} {pick_back}")

        if pick_back:
            self.arm._arm.set_servo_angle(angle=self.place_back_pose, speed=self.servo_speed, wait=True)

        time.sleep(2)
        # open the gripper
        self.arm.open_gripper()
        next_gripper_pos = self.arm.get_gripper_state()
        # move the end effector, parameters: qpos
        quat = mat2quat(action_parameters[:3, :3])
        tar_xyz = (action_parameters[:3, 3] * 1000)
        tar_rpy = quat2rpy(quat)
        target_pose = list(tar_xyz) + list(tar_rpy)

        self.arm.move_to_pose(target_pose, speed=speed, wait=True)

        while True:
            self.arm.set_gripper_state(next_gripper_pos)
            curr_gripper_pose = self.arm.get_gripper_state()
            # Check if the gripper has encountered resistance
            if next_gripper_pos <= 0 or next_gripper_pos - curr_gripper_pose > 10:
                print(f"Gripper stopped closing at position: {curr_gripper_pose}")
                break  # Stop closing if fully closed or resistance is detected  

            # Gradually close the gripper by decreasing the position
            if next_gripper_pos > 0:
                next_gripper_pos -= 100  # Reduce the gripper opening by 50 each step
            else:
                next_gripper_pos = 0  # Make sure the gripper doesn't go below 0

        if pick_back:
            # after picked object, vertical up
            self.arm._arm.set_servo_angle(
                angle=self.place_back_pose, speed=self.servo_speed, is_radian=False, wait=True
            )
        else:
            self.arm._arm.set_servo_angle(
                angle=self.see_poses[0], speed=self.servo_speed, is_radian=False, wait=True
            )

        self.arm.to_back_safty_pose()
        time.sleep(2)

        if not pick_back:
            # place picked object to robot container
            self.arm._arm.set_servo_angle(
                angle=self.place_back_pose, speed=self.servo_speed, is_radian=False, wait=True
            )

            self.arm.open_gripper()
            time.sleep(1)

            # Move the arm to its highest point for safety
            self.arm._arm.set_servo_angle(
                angle=self.see_poses[0], speed=self.servo_speed, is_radian=False, wait=True
            )
            time.sleep(1)

        # arm move to init state
        self.arm.reset()


    def place(
        self, action_parameters=[], speed=50
    ):
        print(f"Running action: place with parameters {action_parameters}")

        # move the end effector, parameters: qpos
        quat = mat2quat(action_parameters[:3, :3])
        xyz = action_parameters[:3, 3] * 1000
        rpy = quat2rpy(quat)
        pose = list(xyz) + list(rpy)
        # Move the arm to its highest point for safety
        # self.arm._arm.set_servo_angle(
        #     angle=self.see_poses[0], speed=self.servo_speed, is_radian=False, wait=True
        # )
        # time.sleep(2)
        self.arm.move_to_pose(pose, wait=True, ignore_error=True, speed=speed)
        # open the gripper
        self.arm.open_gripper()
        time.sleep(1)
        # Move the arm to its highest point for safety
        self.arm._arm.set_servo_angle(
            angle=self.see_poses[0], speed=self.servo_speed, is_radian=False, wait=True
        )
        time.sleep(1)
        # arm move to init state
        self.arm.reset()

    def motion_moving(self, paths: list):
        self.base.motion(paths=paths)

if __name__ == "__main__":
    robotcontrol = RobotControl()
    robotcontrol.arm._arm.set_servo_angle(angle=robotcontrol.place_back_pose, speed=robotcontrol.servo_speed, wait=True)
    # observations = robotcontrol.get_observations(just_wrist=True)
    # action_parameters = np.array([[ 0.59424198,  0.79360074, -0.13066885,  0.53043765],
    #    [-0.77508342,  0.60843885,  0.17043437, -0.10454562],
    #    [-0.21476084,  0.        , -0.97666669, -0.04290964],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # robotcontrol.pick_up(action_parameters=action_parameters)