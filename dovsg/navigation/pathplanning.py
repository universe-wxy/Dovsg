from dovsg.navigation.astar_planner import AStarPlanner
from dovsg.navigation.occupancy_map import occupancy_map
from dovsg.memory.view_dataset import ViewDataset
# from dovisg.navigation.instance_process import MapObjectList
from dovsg.navigation.visualizations import visualize
from pathlib import Path
import torch
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
# plt.ioff()
import matplotlib.colors as mcolors
from dovsg.navigation.map import Map

class PathPlanning:
    def __init__(
        self,
        view_dataset: ViewDataset,
        memory_dir: Path,
        resolution: float=0.05, 
        occ_avoid_radius: float=0.4, 
        min_height: float=0.2,
        conservative: bool=False,
        occ_threshold: int=100,
        pointcloud_visualization: bool=True,
    ):
        self.memory_dir = memory_dir
        self.resolution = resolution
        self.occ_avoid_radius = occ_avoid_radius
        # self.clear_around_bot_radius = clear_around_bot_radius
        self.min_height = min_height  # floor
        self.max_height = self.min_height + 1.2  # round(robot max height)
        self.conservative = conservative
        self.occ_threshold = occ_threshold
        self.pointcloud_visualization = pointcloud_visualization

        self.pcd = view_dataset.index_to_pcd(list(view_dataset.indexes_colors_mapping_dict.keys()))

        self.occ_map = occupancy_map(
            view_dataset=view_dataset,
            min_height=self.min_height,
            max_height=self.max_height,
            resolution=self.resolution,
            occ_threshold=self.occ_threshold,
            conservative=self.conservative,
            occ_avoid=int(np.ceil((self.occ_avoid_radius) / self.resolution))
        )

        # just process x,y coord
        self.astarplanner = AStarPlanner(
            memory_dir=self.memory_dir,
            occ_map=self.occ_map,
            resolution=self.resolution,
            occ_avoid_radius=self.occ_avoid_radius,
            conservative=self.conservative,
        )

        # process x,y,z coor and semantic feature
        # self.instancelocalizer = instance_localizer

        original_point = self.occ_map.origin
        ycells, xcells = self.occ_map.grid.shape

        self.min_x, self.min_y = original_point
        self.max_x, self.max_y = self.min_x + xcells * self.resolution, self.min_y + ycells * self.resolution

        print("\n\n Data process over! \n\n")


    def generate_paths(self, start_xyz, end_xyz, visualize=False):
        end_xy = end_xyz[:2]

        try:
            print("====> A* planning.")
            paths = self.astarplanner.plan(
                start_xy=start_xyz[:2], end_xy=end_xy, remove_line_of_sight_points=True
            )
        except:
            # Sometimes, start_xy might be an occupied obstacle point, in this case, A* is going to throw an error
            # In this case, we will throw an error and still visualize
            print(
                'A* planner said that your robot stands on an occupied point,\n\
                it might be either your hector slam is not tracking robot current position,\n\
                or your min_height or max_height is set to incorrect value so obstacle map is not accurate!'
            )
            paths = None

        # if paths:
            # self.visualization(start_xy, A, B, paths, end_xyz)
        if visualize:
            self.visualization(start_xyz, paths, end_xyz)

        return paths
    
    def visualization(self, start_xy, paths, end_xyz):
        if self.pointcloud_visualization:
            visualize(self.pcd, paths, end_xyz, self.min_height)

        w, h = self.occ_map.grid[::-1].shape
        # fig, ax = plt.subplots(figsize=(16, 16))
        size = 16
        fig, ax = plt.subplots(figsize=(size, int(size * h / w)), facecolor='black')
        # Draw paths only when path planning is done successfully
        if paths:
            xs, ys, thetas = zip(*paths)

        cmap = mcolors.ListedColormap(['green', 'orange'])
        bounds = [0, 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        cbar_img = ax.imshow(self.occ_map.grid[::-1], 
                             extent=(self.min_x, self.max_x, self.min_y, self.max_y), cmap=cmap, norm=norm)
        cbar = plt.colorbar(cbar_img, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(['False', 'True'])
        # occ_map_figure = self.occ_map.grid[::-1].copy()
        # occ_point = np.array(np.where(occ_map_figure == True))
        # free_point = np.array(np.where(occ_map_figure == False))
        # ax.scatter(occ_point[0, :], occ_point[1, :], s=50, c="orange")
        # ax.scatter(free_point[0, :], free_point[1, :], s=50, c="green")
        if paths:
            # ax.plot(xs, ys, c="r")
            ax.scatter(start_xy[0], start_xy[1], s=100, c="r")
            ax.scatter(xs, ys, c="cyan", s=20)
            ax.scatter(end_xyz[0], end_xyz[1], s=100, c="b")
        else:
            ax.scatter(start_xy[0], start_xy[1], s=100, c="r")
            ax.scatter(end_xyz[0], end_xyz[1], s=100, c="b")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        output_path = self.memory_dir / f"navigation_vis.jpg"
        print(output_path)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()