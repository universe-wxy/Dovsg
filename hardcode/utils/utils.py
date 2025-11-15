import numpy as np
import matplotlib.pyplot as plt
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import quat2euler, mat2euler, euler2quat, euler2mat
import open3d as o3d
import plotly.graph_objs as go


# Calculate the transformation matrix through the lookAt logic
# Borrow the logic from https://sapien.ucsd.edu/docs/latest/tutorial/rendering/camera.html#create-and-mount-a-camera
def get_pose_look_at(
    eye, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0])
):
    # Convert to numpy array
    if type(eye) is list:
        eye = np.array(eye)
    if type(target) is list:
        target = np.array(target)
    if type(up) is list:
        up = np.array(up)
    up /= np.linalg.norm(up)
    # Calcualte the rotation matrix
    front = target - eye
    front /= np.linalg.norm(front)
    left = np.cross(up, front)
    left = left / np.linalg.norm(left)
    new_up = np.cross(front, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([front, left, new_up], axis=1)
    mat44[:3, 3] = eye
    return mat44


# The function is to calculate the qpos of the end effector given the front and up vector
# front is defined as the direction of the gripper
# Up is defined as the reverse direction with special shape
def get_pose_from_front_up_end_effector(front, up):
    # Convert to numpy array
    if type(front) is list:
        front = np.array(front)
    if type(up) is list:
        up = np.array(up)
    front = front / np.linalg.norm(front)
    up = -up / np.linalg.norm(up)
    left = np.cross(up, front)
    left = left / np.linalg.norm(left)
    new_up = np.cross(front, left)

    rotation_mat = np.eye(3)
    rotation_mat[:3, :3] = np.stack([left, new_up, front], axis=1)
    quat = mat2quat(rotation_mat)

    return quat


def rpy_to_rotation_matrix(roll, pitch, yaw):
    R = euler2mat(
        roll / 180 * np.pi, pitch / 180 * np.pi, yaw / 180 * np.pi, axes="sxyz"
    )
    return R

def quat2rpy(quat):
    # Convert to numpy array
    if type(quat) is list:
        quat = np.array(quat)
    # Convert to rpy
    rpy = np.array(quat2euler(quat, axes="sxyz")) / np.pi * 180
    return rpy


wrist_serial_number = "347622073075"
top_serial_number = "327522300402"

calib_result_path = "calib/calibrate.pkl"