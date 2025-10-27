# -*- coding: utf-8 -*-
"""
Interactive Depth Map Viewer (Windows Edition)
Author: Hu Wenhui (USTC) + GPT-5 Optimized
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

# ======== 保证使用交互后端（Windows默认TkAgg） ========
matplotlib.use("TkAgg")

# ======== 修改为你的文件路径 ========
path = "/home/universe/workspace/DovSG/third_party/anygrasp_sdk/grasp_detection/example_data/depth.png"
assert os.path.exists(path), f"❌ File not found: {path}"

# ======== 读取深度图 ========
depth = np.load(path)
print(f"✅ Loaded depth map: {depth.shape}, dtype={depth.dtype}")

# 提取有效像素
valid = depth[np.isfinite(depth) & (depth > 0)]
if valid.size == 0:
    raise ValueError("❌ No valid depth pixels found!")

# 自动对比度拉伸范围
vmin, vmax = np.percentile(valid, [1, 99])
print(f"📊 Depth range (1%-99%): {vmin:.3f} ~ {vmax:.3f} m")

# ======== 绘制窗口 ========
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
plt.suptitle(f"Depth Map Viewer\n{path}", fontsize=11)

# 左：深度图像
im = ax[0].imshow(np.where(depth > 0, depth, np.nan),
                  cmap='turbo', vmin=vmin, vmax=vmax)
ax[0].set_title("Depth Visualization (m)")
ax[0].axis("off")
cb = plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
cb.set_label("Depth (meters)")

# 右：深度分布直方图
ax[1].hist(valid, bins=100, color="royalblue", alpha=0.8)
ax[1].set_title("Depth Histogram")
ax[1].set_xlabel("Depth (m)")
ax[1].set_ylabel("Pixel Count")

# ======== 鼠标悬停显示深度 ========
def format_coord(x, y):
    h, w = depth.shape
    ix, iy = int(x + 0.5), int(y + 0.5)
    if 0 <= ix < w and 0 <= iy < h:
        val = depth[iy, ix]
        return f"x={ix}, y={iy}, depth={val:.4f} m" if np.isfinite(val) and val > 0 else f"x={ix}, y={iy}, depth=NaN"
    return f"x={ix}, y={iy}"

ax[0].format_coord = format_coord

# ======== 按键保存功能 ========
def on_key(event):
    if event.key.lower() == 's':
        save_path = os.path.splitext(path)[0] + "_viz.png"
        plt.savefig(save_path, dpi=200)
        print(f"💾 Saved visualization to: {save_path}")

fig.canvas.mpl_connect("key_press_event", on_key)

plt.tight_layout()
plt.show()