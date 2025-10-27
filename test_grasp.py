# -*- coding: utf-8 -*-
"""
test_grasp.py —— 等价于 Controller.pick_up(object1, observations, is_visualize=True)
仅执行 GraspNet 抓取预测，可视化由 ObjectHandler 内部控制。
"""

import torch
from dovsg.controller import Controller
from dovsg.manipulation.objecthandler import ObjectHandler

def pick_up(object1: str, observations: dict, is_visualize=True):
    """从原 Controller.pick_up 改写，移除 self/socket，只做预测"""
    print(f"Running Pick up({object1}) Task.")

    # === 初始化 GraspNet 处理器 ===
    object_handler = ObjectHandler(
        box_threshold=0.6,
        text_threshold=0.6,
        device="cuda",
        is_visualize=is_visualize
    )

    # === 调用 GraspNet ===
    target_pose = object_handler.pickup(
        observations=observations,
        query=object1
    )

    # === 打印结果 ===
    print("\n===== GraspNet Prediction Result =====")
    if isinstance(target_pose, dict):
        for k, v in target_pose.items():
            if hasattr(v, "shape"):
                print(f"{k}: shape={v.shape}")
            else:
                print(f"{k}: {v}")
    else:
        print(target_pose)

    del object_handler
    torch.cuda.empty_cache()
    print("[SUCCESS] GraspNet test finished!")


def main():
    """直接执行抓取预测测试"""
    object1 = "red pepper"

    # 仅用于取当前观测（RGBD + 点云）
    controller = Controller(
        step=0,
        tags="room1",
        save_memory=True,
        debug=True
    )

    observations, ok = controller.get_align_observations(
        just_wrist=True,
        show_align=True,
        use_inlier_mask=False,
        self_align=False,
        align_to_world=False,
        save_name="grasp_net"
    )

    if not ok:
        raise RuntimeError("Alignment failed, 请检查 ACE 模型或相机配置。")

    pick_up(object1=object1, observations=observations, is_visualize=True)


if __name__ == "__main__":
    main()
