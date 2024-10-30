from dovsg.memory.view_dataset import ViewDataset
from dovsg.navigation.bounds import Bounds
from dovsg.navigation.map import Map
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from dataclasses import dataclass

def occupancy_map(
    view_dataset: ViewDataset,
    min_height: float,
    max_height: float,
    resolution: float=0.01,
    occ_threshold: int=100,
    conservative: bool=False,
    occ_avoid: int=2
): 

    bounds = view_dataset.bounds
    points = view_dataset.index_to_point(list(view_dataset.indexes_colors_mapping_dict.keys()))
    origin = (bounds.xmin, bounds.ymin)

    xbins, ybins = int(bounds.xdiff / resolution) + 2, int(bounds.ydiff / resolution) + 2
    counts = np.zeros((ybins, xbins), dtype=np.int32).flatten()
    any_counts = np.zeros((ybins, xbins), dtype=np.int32).flatten()

    xs = np.floor(((points[:, 0] - origin[0] + resolution / 2) / resolution)).astype(np.int32)
    ys = np.floor(((points[:, 1] - origin[1] + resolution / 2) / resolution)).astype(np.int32)

    # Counts the number of occupying points in each cell.
    occ_xys = np.logical_and(points[:, 2] >= min_height, points[:, 2] <= max_height)
    occ_inds = ys[occ_xys] * xbins + xs[occ_xys]
    np.add.at(counts, occ_inds, 1)

    # Keeps track of the cells that have any points from anywhere.
    inds = ys * xbins + xs
    inds = np.clip(inds, 0, any_counts.size - 1)
    np.add.at(any_counts, inds, 1)

    # assert counts is not None and any_counts is not None, "No points in the dataset"
    assert counts is not None, "No points in the dataset"
    counts = counts.reshape((ybins, xbins))
    any_counts = any_counts.reshape((ybins, xbins))


    occ_map = (counts >= occ_threshold)
    occ_map_copy = occ_map.copy()

    for i in range(occ_map.shape[0]):
        for j in range(occ_map.shape[1]):
            if occ_map_copy[i, j]:
                occ_map[
                    max(0, i - occ_avoid): min(occ_map.shape[0] - 1, i + occ_avoid), 
                    max(0, j - occ_avoid): min(occ_map.shape[1] - 1, j + occ_avoid)
                ] = True

    if conservative:
        no_counts = (any_counts == 0)
        occ_map = np.logical_or(occ_map, no_counts)
        # occ_map = (occ_map | ~any_counts).astype(np.int32)
    else:
        occ_map = occ_map

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 10))
    # occ_map_figure = occ_map.copy()[::-1]
    # occ_point = np.array(np.where(occ_map_figure == True))
    # free_point = np.array(np.where(occ_map_figure == False))
    # plt.scatter(occ_point[0, :], occ_point[1, :], color="red")
    # plt.scatter(free_point[0, :], free_point[1, :], color="green")
    # plt.show()
    # plt.close()
    occupancy_map = Map(occ_map, resolution, origin)


    # import matplotlib.pyplot as plt
    # import numpy as np
    # occ_map_numeric = occ_map.astype(np.int32)
    # plt.figure(figsize=(10, 8))
    # plt.imshow(occ_map_numeric, cmap='hot', interpolation='nearest')
    # plt.colorbar(label='Occupancy')
    # plt.title('Occupancy Map')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.savefig("1.png")

    return occupancy_map

