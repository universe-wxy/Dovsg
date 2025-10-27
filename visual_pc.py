# -*- coding: utf-8 -*-
import os
import numpy as np
import open3d as o3d

# ======== 修改你的文件路径 ========
path = "/home/universe/workspace/DovSG/third_party/anygrasp_sdk/grasp_detection/example_data/depth.png"

assert os.path.isfile(path), f"File not found: {path}"
arr = np.load(path)
print(f"✅ Loaded: shape={arr.shape}, dtype={arr.dtype}")

def as_float64(a):
    return a.astype(np.float64, copy=False)

pcd = o3d.geometry.PointCloud()

if arr.ndim == 3 and arr.shape[2] == 3:
    # 图像式 3 通道：展开为 (H*W, 3)
    pts = arr.reshape(-1, 3)
    pts = as_float64(pts)
    pcd.points = o3d.utility.Vector3dVector(pts)
    print(f"🔧 Interpreted as per-pixel 3D (or 3-channel) → points: {pts.shape}")
elif arr.ndim == 2 and arr.shape[1] == 3:
    # (N,3) 直接作为 XYZ
    pts = as_float64(arr[:, :3])
    pcd.points = o3d.utility.Vector3dVector(pts)
    print(f"🔧 Interpreted as Nx3 XYZ → points: {pts.shape}")
elif arr.ndim == 2 and arr.shape[1] >= 6:
    # (N,6)：XYZ + RGB
    pts = as_float64(arr[:, :3])
    pcd.points = o3d.utility.Vector3dVector(pts)
    cols = arr[:, 3:6]
    # 颜色归一化到 [0,1]
    if cols.max() > 1.0:
        cols = cols / 255.0
    cols = np.clip(cols, 0.0, 1.0).astype(np.float64, copy=False)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    print(f"🔧 Interpreted as Nx6 (XYZ + RGB) → points: {pts.shape}, colors: {cols.shape}")
else:
    raise ValueError(
        f"Unsupported array shape {arr.shape}. "
        f"Expect (H,W,3), (N,3), or (N,6). "
        f"如果是深度图 (H,W) 需要用相机内参反投影成点云。"
    )

# 可选：体素降采样（速度更快，窗口更流畅），按需开启
# pcd = pcd.voxel_down_sample(voxel_size=0.01)

# 加一个坐标系帮助观察
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

# 计算法线（可选）
if len(pcd.points) > 0:
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

o3d.visualization.draw_geometries([pcd, axes], window_name="3D 点云可视化")