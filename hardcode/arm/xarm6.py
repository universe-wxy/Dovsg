from xarm import version
from xarm.wrapper import XArmAPI
import numpy as np
import time
import traceback


class XARM6:
    def __init__(
        self,
        interface="192.168.1.233",
        # The pose corresponds to the servo angle
        # init_servo_angle=[0, -45, -45, 0, 90, 0],  # [0, -60, -30, 0, 90, 0]
        init_servo_angle=[0, -45, -90, 0, 135, 0],
        back_safty_angle=[180, -45, -90, 0, 135, 0]
    ):
        self.pprint("xArm-Python-SDK Version:{}".format(version.__version__))
        self.alive = True
        self._arm = XArmAPI(interface, baud_checkset=False)
        self.init_servo_angle = init_servo_angle
        self.back_safty_angle = back_safty_angle
        self._robot_init()

    # Robot Init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        self._arm.set_gripper_enable(True)
        self._arm.set_gripper_mode(0)
        self._arm.clean_gripper_error()
        self._arm.set_collision_sensitivity(1)
        # self._arm.set_tcp_load(1.035, [0, 0, 48])
        self._arm.set_tcp_load(1.035, [0, 0, 0.48])
        # self._arm.set_tcp_load(1.035, [0, 0, 0.60])
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(
            self._error_warn_changed_callback
        )
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, "register_count_changed_callback"):
            self._arm.register_count_changed_callback(self._count_changed_callback)
        self.reset()
        self.open_gripper()

    # Robot Contrl: here the pose is the end-effector pose [X, Y, Z, roll, pitch, yaw]
    def move_to_pose(self, pose, wait=True, ignore_error=False, speed=200):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code = self._arm.set_position(
            pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], speed=speed, wait=wait
        )
        if not ignore_error:
            if not self._check_code(code, "set_position"):
                raise ValueError("move_to_pose Error")
        return True

    def get_current_pose(self):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code, pose = self._arm.get_position()
        if not self._check_code(code, "get_position"):
            raise ValueError("get_current_pose Error")
        return pose

    def open_gripper(self, wait=True, half_open=False):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        if half_open:
            code = self._arm.set_gripper_position(460, wait=wait)
        else:
            code = self._arm.set_gripper_position(830, wait=wait)
        if not self._check_code(code, "set_gripper_position"):
            raise ValueError("open_gripper Error")
        return True

    def close_gripper(self, wait=True):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code = self._arm.set_gripper_position(0, wait=wait)
        if not self._check_code(code, "set_gripper_position"):
            raise ValueError("close_gripper Error")
        return True

    def set_gripper_state(self, gripper_pose: int, wait=True):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code = self._arm.set_gripper_position(gripper_pose, wait=wait)
        if not self._check_code(code, "set_gripper_position"):
            raise ValueError("close_gripper Error")
        return True
    
    def get_gripper_state(self):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code, state = self._arm.get_gripper_position()
        if not self._check_code(code, "get_gripper_position"):
            raise ValueError("get_gripper_position Error")
        return state

    def to_back_safty_pose(self):
        self._arm.set_servo_angle(
            angle=self.back_safty_angle, speed=20, is_radian=False, wait=True
        )   

    def reset(self):
        # This can proimise the initial position has the correct joint angle
        servo_pose = self._arm.get_servo_angle()
        if servo_pose[1][0] > 135 or servo_pose[1][0] < -135:
            # self._arm.set_servo_angle(
            #     angle=[180, -45, -90, 0, 135, 0], speed=20, is_radian=False, wait=True
            # )
            self.to_back_safty_pose()
            
        time.sleep(1)
        self._arm.set_servo_angle(
            angle=self.init_servo_angle, is_radian=False, wait=True
        )

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data["error_code"] != 0:
            self.alive = False
            self.pprint("err={}, quit".format(data["error_code"]))
            self._arm.release_error_warn_changed_callback(
                self._error_warn_changed_callback
            )

    # Register state changed callback
    def _state_changed_callback(self, data):
        if data and data["state"] == 4:
            self.alive = False
            self.pprint("state=4, quit")
            self._arm.release_state_changed_callback(self._state_changed_callback)

    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint("counter val: {}".format(data["count"]))

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint(
                "{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}".format(
                    label,
                    code,
                    self._arm.connected,
                    self._arm.state,
                    self._arm.error_code,
                    ret1,
                    ret2,
                )
            )
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print(
                "[{}][{}] {}".format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                    stack_tuple[1],
                    " ".join(map(str, args)),
                )
            )
        except:
            print(*args, **kwargs)

    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False


if __name__ == "__main__":
    xarm = XARM6()
    # pose must be [X, Y, Z, roll, pitch, yaw]
    # poses = [
    #     # Pose 1
    #     [263.731323, 26.548101, 387.849731, -173.282892, 23.043217, -10.037705],
    #     [268.941742, -11.555555, 410.964691, -175.391262, 31.465467, -10.371338],
    #     [277.991882, -10.138917, 429.85141, -173.874184, 18.252946, 6.556585],
    #     [263.773407, -5.113694, 424.882446, -172.188485, 25.204127, -3.061944],
    #     [286.275574, 0.48078, 388.695129, 169.20475, 19.140916, 0.776644],
    #     [268.292938, 12.322101, 396.480072, -179.760409, 25.735603, 5.932634],
    #     # Pose 2
    #     [363.941864, 165.311935, 365.18042, 170.33531, 23.462622, -89.093734],
    #     [361.088531, 163.330643, 361.326599, 175.971153, 25.606, -84.645118],
    #     [385.088837, 166.307434, 358.266205, 164.065376, 20.410017, -86.535477],
    #     [375.216125, 177.15918, 369.692169, 174.871532, 23.472076, -90.972119],
    #     [363.056671, 151.710983, 356.220184, 169.952632, 20.056273, -95.139126],
    #     [342.381866, 146.086578, 347.81427, 167.615021, 24.641712, -101.520616],
    #     # Pose 3
    #     [318.695923, -183.728088, 377.404907, -168.434752, 26.44057, 50.166122],
    #     [319.03186, -172.396683, 373.445709, -178.302919, 28.056712, 49.804929],
    #     [321.078705, -172.923706, 399.83429, -164.069673, 33.00701, 45.33626],
    #     [305.544403, -179.392273, 360.520477, -165.835185, 25.209054, 52.448843],
    #     [303.150604, -200.736572, 366.663544, 179.644958, 19.041909, 46.242622],
    #     [328.160797, -174.039917, 394.734589, -177.424575, 22.076923, 52.370233],
    # ]
    # for pose in poses:
    #     # Move to the pose and wait for 5s to make it stable
    #     xarm.move_to_pose(pose=pose, wait=True, ignore_error=True)