# -*- coding: utf-8 -*-
# 从 XML 读取相机位姿 -> 写外参 4x4 到 poses_droidslam
# 读取相机系 PLY -> 按外参变到世界系 -> 每帧 world_ply 与 merged_world.ply
import math, re
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
import open3d as o3d

# ========= 路径配置 =========
XML_PATH = Path("/home/universe/workspace/DovSG/simulation/room5.xml")
BASE_DIR = Path("/home/universe/workspace/DovSG/data_example/room5")   # ← 指向 room5 目录

PLY_DIR        = BASE_DIR / "ply"                         # 相机系点云输入
POSES_DIR      = BASE_DIR / "poses_droidslam"             # 外参输出
WORLD_OUT_DIR  = BASE_DIR / "world_ply"                   # 每帧世界系输出
MERGED_OUT_PLY = BASE_DIR / "merged_world.ply"            # 融合输出
VOXEL = 0.01  # 体素下采样，0 表示不启用




# ========= 旋转/外参工具 =========
def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)
def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)
def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)
def euler_xyz_to_R(ex, ey, ez):
    # MuJoCo: euler="x y z"（弧度）→ R = Rx @ Ry @ Rz（列向量约定）
    return Rx(ex) @ Ry(ey) @ Rz(ez)

# OpenCV 相机(+X右,+Y下,+Z前) → MuJoCo/OpenGL 相机(+X右,+Y上,-Z前)
R_CFIX = np.diag([1.0, -1.0, -1.0])

def build_T_world_cam(pos, euler_xyz):
    R_wb = euler_xyz_to_R(*euler_xyz)  # body(OpenGL)→world
    R_wc = R_wb @ R_CFIX               # cam(OpenCV)→body(OpenGL)→world
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_wc
    T[:3,  3] = np.asarray(pos, dtype=np.float64)
    return T

# ========= XML 解析 =========
def parse_dataset_cameras(xml_path: Path):
    """
    解析 <body name="dataset_cameras"> ... </body> 下的子 body：
      <body name="cam_ds_001" pos="x y z" euler="ex ey ez"><camera name="ds_001" .../></body>
    返回 dict: { "000000": {"pos":[x,y,z], "euler":[ex,ey,ez]} , ... }
    规则：ds_001 → 000000.txt
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # find the body with name="dataset_cameras"
    dataset_node = None
    for body in root.iter("body"):
        if body.get("name") == "dataset_cameras":
            dataset_node = body
            break
    if dataset_node is None:
        raise RuntimeError('未找到 <body name="dataset_cameras"> 节点')

    poses = {}
    # 只遍历 dataset_cameras 下的直接子 body（或也可递归 .iter('body') 视你的层级）
    for cam_body in list(dataset_node):
        if cam_body.tag != "body":
            continue
        # 必须包含一个 <camera name="ds_###"/>
        cam = cam_body.find("camera")
        if cam is None:
            continue
        cam_name = cam.get("name", "")
        m = re.match(r"ds_(\d+)$", cam_name)
        if not m:
            continue
        ds_id = int(m.group(1))
        stem = f"{ds_id-1:06d}"  # ds_001 -> 000000

        pos_str   = cam_body.get("pos",   "").strip()
        euler_str = cam_body.get("euler", "").strip()
        if not pos_str or not euler_str:
            print(f"[WARN] {cam_body.get('name','(no-name)')} 缺少 pos/euler，跳过")
            continue

        pos   = [float(x) for x in pos_str.split()]
        euler = [float(x) for x in euler_str.split()]  # 弧度
        poses[stem] = {"pos": pos, "euler": euler}

    if not poses:
        raise RuntimeError("dataset_cameras 下未解析到任何相机位姿（检查 XML 格式）")
    print(f"[INFO] 解析到 {len(poses)} 个相机位姿")
    return poses

# ========= 主流程 =========
def main():
    PLY_DIR.mkdir(parents=True, exist_ok=True)
    POSES_DIR.mkdir(parents=True, exist_ok=True)
    WORLD_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 解析 XML 得到位姿
    poses = parse_dataset_cameras(XML_PATH)

    # 2) 写外参 4×4 到 poses_droidslam/000000.txt …
    for stem in sorted(poses.keys()):
        T_wc = build_T_world_cam(poses[stem]["pos"], poses[stem]["euler"])
        np.savetxt(POSES_DIR / f"{stem}.txt", T_wc, fmt="%.8f")
    print(f"[INFO] 外参已写入: {POSES_DIR}")

    # 3) 读取相机系 PLY、变为世界系并保存，顺带融合
    ply_paths = sorted(PLY_DIR.glob("*.ply"))
    if not ply_paths:
        raise RuntimeError(f"{PLY_DIR} 下没有 *.ply（相机系点云）")

    merged = o3d.geometry.PointCloud()
    total_in_pts = 0
    converted_cnt = 0

    for ply in ply_paths:
        stem = ply.stem  # "000000" ...
        if stem not in poses:
            print(f"[WARN] 位姿缺失，跳过 {ply.name}")
            continue

        pcd = o3d.io.read_point_cloud(str(ply))
        if pcd.is_empty():
            print(f"[WARN] 空点云，跳过 {ply.name}")
            continue

        T_wc = build_T_world_cam(poses[stem]["pos"], poses[stem]["euler"])
        pcd_world = o3d.geometry.PointCloud(pcd)  # 克隆
        pcd_world.transform(T_wc)

        # 每帧世界系输出
        out_ply = WORLD_OUT_DIR / f"{stem}_world.ply"
        o3d.io.write_point_cloud(str(out_ply), pcd_world, write_ascii=False)

        # 融合（简单相加）
        merged += pcd_world
        total_in_pts += len(pcd.points)
        converted_cnt += 1
        print(f"[OK] {ply.name} → world  (pts={len(pcd.points)})")

    if converted_cnt == 0:
        print("[ERR] 没有可融合的点云"); return

    print(f"[INFO] 融合前点数: {len(merged.points)}（输入合计≈{total_in_pts}）")
    if VOXEL and VOXEL > 0:
        merged = merged.voxel_down_sample(VOXEL)
        print(f"[INFO] 体素下采样({VOXEL} m)后: {len(merged.points)}")

    o3d.io.write_point_cloud(str(MERGED_OUT_PLY), merged, write_ascii=False)
    print(f"[DONE] 已保存融合点云: {MERGED_OUT_PLY}")

if __name__ == "__main__":
    main()
