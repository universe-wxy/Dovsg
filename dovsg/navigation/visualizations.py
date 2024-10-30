import math
from tqdm import tqdm
import open3d as o3d
import numpy as np
import os
from pathlib import Path

def generate_colors(num_points):
    colors = []
    for i in range(num_points):
        t = i / (num_points - 1)
        color = [1 - t, 0, t]
        colors.append(color)
    return colors

def create_dashed_cylinder_line(points, radius=0.03, dash_length=0.07, gap_length=0.03):
    colors = generate_colors(len(points))
    geometries = []
    for i in range(len(points) - 1):
        start_point = np.array(points[i])
        end_point = np.array(points[i + 1])
        start_color = colors[i]
        end_color = colors[i + 1]
        vec = end_point - start_point
        seg_length = np.linalg.norm(vec)
        vec_normalized = vec / seg_length
        n_dashes = math.ceil(seg_length / (dash_length + gap_length))

        for j in range(n_dashes):
            new_dash_length = min(dash_length, seg_length - j * (dash_length + gap_length))
            dash_start = start_point + vec_normalized * j * (new_dash_length + gap_length)
            dash_end = dash_start + vec_normalized * new_dash_length
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=new_dash_length)
            cylinder.translate((dash_start + dash_end) / 2)

            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, vec)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.dot(z_axis, vec_normalized))
            rotation_matrix = cylinder.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
            cylinder.rotate(rotation_matrix, center=dash_start)

            # Interpolate color based on position
            t = j / n_dashes
            interpolated_color = (1 - t) * np.array(start_color) + t * np.array(end_color)
            cylinder.paint_uniform_color(interpolated_color)
            geometries.append(cylinder)
    
    return geometries

def add_arrows_to_line(line_set, arrow_length=0.2, cylinder_radius=0.02, cone_radius=0.05):
    arrows = []

    for line in line_set.lines:
        start_idx, end_idx = line
        start_point = np.asarray(line_set.points[start_idx])
        end_point = np.asarray(line_set.points[end_idx])
        cylinder, cone = create_arrow_geometry(start_point, end_point, arrow_length, cylinder_radius, cone_radius)
        arrows.append(cylinder)
        arrows.append(cone)

    return arrows

def create_arrow_geometry(start_point, end_point, arrow_length=0.2, cylinder_radius=0.02, cone_radius=0.05):
    vec = end_point - start_point
    vec_norm = np.linalg.norm(vec)
    arrow_vec = vec / vec_norm * arrow_length

    # Cylinder (arrow's body)
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=arrow_length)
    cylinder.translate(start_point)
    cylinder.rotate(cylinder.get_rotation_matrix_from_xyz([np.arccos(arrow_vec[2] / np.linalg.norm(arrow_vec)), 0, np.arctan2(arrow_vec[1], arrow_vec[0])]), center=start_point)

    # Cone (arrow's head)
    cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=arrow_length)
    cone.translate(start_point + vec - arrow_vec)
    cone.rotate(cone.get_rotation_matrix_from_xyz([np.arccos(arrow_vec[2] / np.linalg.norm(arrow_vec)), 0, np.arctan2(arrow_vec[1], arrow_vec[0])]), center=start_point + vec - arrow_vec)
    
    return cylinder, cone

# def vis_voxelized_pointcloud(global_points, masks, colors):
#     pcd_all = o3d.geometry.PointCloud()
#     for point, mask, color in tqdm(zip(global_points, masks, colors), 
#                                    total=len(global_points), desc="Point Clouds"):
#         xyz = point[mask]
#         rgb = color[mask]
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(xyz)
#         pcd.colors = o3d.utility.Vector3dVector(rgb)
#         pcd_all += pcd
#     return pcd_all

# def visualize(path, end_xyz, pointcloud_path, view_dataset, min_height, max_height):
def visualize(pcd, path, end_xyz, min_height):
    
    if path is not None:
        path = np.array(np.array(path).tolist())
        print(path)
        start_point = path[0, :]
    end_point = np.array(end_xyz)

    if path is not None:
        path[:, 2] = min_height
        lines = create_dashed_cylinder_line(path)
    # end_point[2] = (min_height + max_height) / 2

    # Create spheres for start and end points
    if path is not None:
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)

    # Set the position of the spheres
    if path is not None:
        start_sphere.translate(start_point)
    end_sphere.translate(end_point)

    # Set different colors for clarity
    # lines.paint_uniform_color([1, 0, 0])  # Red path
    if path is not None:
        start_sphere.paint_uniform_color([0, 1, 0])  # Green start
    end_sphere.paint_uniform_color([1, 0, 0])  # Blue end

    # Visualize
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(visible=True)
    if path is not None:
        geometries = [pcd, *lines, start_sphere, end_sphere]
    else:
        geometries = [pcd, end_sphere]
    visualizer.poll_events()
    visualizer.update_renderer()
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    visualizer.run()
    visualizer.destroy_window()
