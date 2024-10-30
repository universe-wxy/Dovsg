from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    @property
    def xdiff(self) -> float:
        return self.xmax - self.xmin
    @property
    def ydiff(self) -> float:
        return self.ymax - self.ymin
    @property
    def zdiff(self) -> float:
        return self.zmax - self.zmin
    
    @property
    def lower_bound(self) -> np.ndarray:
        return np.array([self.xmin, self.ymin, self.zmin])

    @property
    def higher_bound(self) -> np.ndarray:
        return np.array([self.xmax, self.ymax, self.zmax])

    @classmethod
    def from_arr(cls, bounds) -> "Bounds":
        assert bounds.shape == (3, 2), f"Invalid bounds shape: {bounds.shape}"
        return Bounds(
            xmin=bounds[0, 0].item(),
            xmax=bounds[0, 1].item(),
            ymin=bounds[1, 0].item(),
            ymax=bounds[1, 1].item(),
            zmin=bounds[2, 0].item(),
            zmax=bounds[2, 1].item(),
        )