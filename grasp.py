#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XArm6 动作流程：
1) Z 上移 +200 mm
2) X 前移 +300 mm
3) 平面内旋转 +90°（绕 Z 轴：yaw += 90°）
4) Z 下移 -390 mm  (按你当前设置)
5) 夹爪闭合（数值缩小）
6) Z 上移 +300 mm（夹爪闭合后抬起）
7) 等待键盘回车 -> 回到零位 (reset) 并张开爪
"""

import time
from xarm.wrapper import XArmAPI

ROBOT_IP = '192.168.1.222'
SPEED    = 100.0
ACC      = 1000.0
Z_UP     = 200.0     # +20 cm
X_FWD    = 300.0     # +30 cm
YAW_ROT  = +90.0     # 平面内旋转 +90°
Z_DOWN   = -390.0    # 你当前设置
GRIP_CLOSE = 100     # 闭合位置（越小越紧）
GRIP_OPEN  = 850     # 张开位置
Z_UP_AFTER_GRIP = 300.0  # 夹爪闭合后上移 +30 cm

def norm_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def move_cart(arm, x, y, z, r, p, yw, desc):
    code = arm.set_position(x=x, y=y, z=z, roll=r, pitch=p, yaw=yw,
                            speed=SPEED, mvacc=ACC, is_radian=False, wait=True)
    if code != 0:
        raise RuntimeError(f"{desc} 失败，code={code}")

def main():
    print("🦾 正在连接 XArm6 ...")
    arm = XArmAPI(ROBOT_IP); arm.connect()
    print("✅ 已连接:", ROBOT_IP)

    # 初始化
    arm.clean_error(); arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0); arm.set_state(0)
    time.sleep(0.5)

    # 启用夹爪并先张开
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_position(GRIP_OPEN, wait=True)
    print("🤲 夹爪已打开")

    # 当前位姿
    code, pose = arm.get_position(is_radian=False)
    if code != 0:
        print(f"❌ 获取位姿失败，code={code}")
        arm.disconnect(); return
    x, y, z, r, p, yw = pose
    print("当前位置:", pose)

    try:
        # 1) 上移 +20 cm
        tz = z + Z_UP
        print(f"⬆️ Z 上移 {Z_UP} mm → z={tz:.1f}")
        move_cart(arm, x, y, tz, r, p, yw, "上移")

        # 2) 前移 +30 cm
        tx = x + X_FWD
        print(f"👉 X 前移 {X_FWD} mm → x={tx:.1f}")
        move_cart(arm, tx, y, tz, r, p, yw, "前移")

        # 3) 平面内旋转 +90°（yaw）
        new_yaw = norm_deg(yw + YAW_ROT)
        print(f"🧭 平面内旋转 {YAW_ROT:+.1f}° → yaw={new_yaw:.1f}")
        move_cart(arm, tx, y, tz, r, p, new_yaw, "旋转 yaw")

        # 4) 下移（按你当前设置 -390mm）
        tz2 = tz + Z_DOWN
        print(f"⬇️ Z 下移 {abs(Z_DOWN)} mm → z={tz2:.1f}")
        move_cart(arm, tx, y, tz2, r, p, new_yaw, "下移")

        # 5) 夹爪闭合
        print(f"✋ 夹爪闭合至 {GRIP_CLOSE}")
        arm.set_gripper_position(GRIP_CLOSE, wait=True)

        # 6) 夹爪闭合后上移 +30 cm
        tz3 = tz2 + Z_UP_AFTER_GRIP
        print(f"⬆️ 闭合后上移 {Z_UP_AFTER_GRIP} mm → z={tz3:.1f}")
        move_cart(arm, tx, y, tz3, r, p, new_yaw, "闭合后上移")

        # 7) 等待键盘输入再归零
        input("⏸ 已抬起。按回车键继续归零（Ctrl+C 取消）... ")

        print("↩️ 正在归零...")
        code = arm.reset(wait=True)
        if code == 0:
            print("✅ 已回到零位")
        else:
            print(f"⚠️ 归零失败，code={code}")

        # 归零后张开爪
        arm.set_gripper_position(GRIP_OPEN, wait=True)
        print("🤲 已重新张开夹爪")

    except KeyboardInterrupt:
        print("\n🛑 已取消归零。保持当前位姿。")
    except Exception as e:
        print(f"❌ 出错：{e}")
    finally:
        arm.disconnect(); print("🔌 已断开连接。")

if __name__ == "__main__":
    main()
