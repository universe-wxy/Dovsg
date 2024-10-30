from dataclasses import dataclass
import numpy as np

@dataclass
class Map:
    grid: np.ndarray
    resolution: float
    origin: tuple[float, float]
    def to_pt(self, xy: tuple[float, float]) -> tuple[int, int]:
        return (
            int((xy[0] - self.origin[0]) / self.resolution + 0.5),
            int((xy[1] - self.origin[1]) / self.resolution + 0.5),
        )
    def to_xy(self, pt: tuple[int, int]) -> tuple[float, float]:
        return (
            pt[0] * self.resolution + self.origin[0],
            pt[1] * self.resolution + self.origin[1],
        )
    def is_occupied(self, pt: tuple[int, int]) -> bool:
        return bool(self.grid[pt[1], pt[0]])