import cv2
import numpy as np
import open3d as o3d
from IPython import embed
from PIL import Image
from utils import RECORDER_DIR, end2cam
from utils import pose_Euler_to_T
from utils import DEPTH_MIN, DEPTH_MAX, DEPTH_SCALE
from utils import transform_to_translation_quaternion
import json
import os
from tqdm import tqdm
import shutil
from IPython import embed

# fx = intrinsics[0, 0]
# fy = intrinsics[1, 1]
# cx = intrinsics[0, 2]
# cy = intrinsics[1, 2]

def apply_pose_to_xyz(xyz: np.ndarray, pose: np.ndarray) -> np.ndarray:
    xyz_reshaped = xyz.reshape(-1, 3)
    xyz_reshaped = np.hstack([xyz_reshaped, np.ones((xyz_reshaped.shape[0], 1))])
    xyz_transformed = np.dot(pose, xyz_reshaped.T).T[..., :3].reshape(xyz.shape)
    return xyz_transformed

def get_xyz(depth, mask, intrinsics):
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["ppx"], intrinsics["ppy"]
    xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points_xyz = np.stack((points_x, points_y, points_z), axis=-1).astype(np.float32)
    points_xyz[mask] = 0.0
    return points_xyz

def rgbd_to_pcd(rgb_image_path, depth_image_path, T_cam_in_base, pcd_path, intrinsics):
    image = np.asarray(Image.open(rgb_image_path), dtype=np.uint8)
    depth = (np.asarray(Image.open(depth_image_path), dtype=np.uint16)) / DEPTH_SCALE
    mask = np.logical_or(depth < DEPTH_MIN, depth > DEPTH_MAX)
    rgb = image / 255
    xyz = get_xyz(depth, mask, intrinsics)
    xyz= apply_pose_to_xyz(xyz, T_cam_in_base)
    mask = (~mask & (np.random.rand(*mask.shape) > 0.))
    rgb, xyz = rgb[mask], xyz[mask]
    return rgb, xyz

def pose_process(pose_end_in_base):
    T_end_in_base = pose_Euler_to_T(pose_end_in_base).astype(np.float32)
    T_cam_in_base = end2cam(T_end_in_base).astype(np.float32)

    print(f"end: {transform_to_translation_quaternion(T_end_in_base)}")
    print(f"cam: {transform_to_translation_quaternion(T_cam_in_base)}")
    print("*******************************************")
    return T_cam_in_base

def main():
    rgb_image_dir = RECORDER_DIR / "r3d_py/color"
    depth_image_dir = RECORDER_DIR / "r3d_py/depth"
    pose_path = RECORDER_DIR / "r3d_py/poses.json"
    pcd_dir = RECORDER_DIR / "r3d_py/pcd"
    intrinsics_path = RECORDER_DIR / "r3d_py/intrinsics.json"

    with open(intrinsics_path, "r") as f:
        intrinsics = json.load(f)

    if pcd_dir.exists():
        shutil.rmtree(pcd_dir)
    pcd_dir.mkdir(parents=True, exist_ok=True)
    with open(pose_path, 'r') as f:
        end_poses = json.load(f)

    color_image_paths = [os.path.join(rgb_image_dir, f) for f in 
                            sorted(os.listdir(rgb_image_dir), key=lambda x: int(os.path.basename(x).split(".")[0]))]
    depth_image_paths = [os.path.join(depth_image_dir, f) for f in 
                            sorted(os.listdir(depth_image_dir), key=lambda x: int(os.path.basename(x).split(".")[0]))]

    xyzs = []
    rgbs = []
    for index in tqdm(range(0, len(color_image_paths), 5), desc="point cloud"):
        img_path = color_image_paths[index]
        depth_path = depth_image_paths[index]
        name = os.path.basename(img_path)
        pcd_path = os.path.join(str(pcd_dir), name.split(".")[0] + ".ply")
        rgb, xyz = rgbd_to_pcd(img_path, depth_path, pose_process(end_poses[name.split(".")[0]]), pcd_path, intrinsics)
        rgbs.append(rgb)
        xyzs.append(xyz)

    xyzs = np.vstack(xyzs)
    rgbs = np.vstack(rgbs)
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(xyzs)
    merged_pcd.colors = o3d.utility.Vector3dVector(rgbs)

    # Flip the pcd
    # embed()
    # merged_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window(visible=True)

    o3d.visualization.draw_geometries([merged_pcd])
    # visualizer.add_geometry(merged_pcd)
    # visualizer.add_geometry(coordinate_frame)
    # visualizer.run()
    # visualizer.destroy_window()

if __name__ == "__main__":
    main()