import zmq
import zlib
from frame_recorder import RecorderImage
from agv_control import AgvController
from hardcode.robotarm.arm_control import ArmController
from utils import AGV_IP, ROBOT_IP
import time
import pickle
import numpy as np
"""receive dict should just has message, poses, paths"""
"""input dict should just involve keys: rgb, depth, pose, calib[fx, fy, ppx, ppy], paths: []"""

class ZmqServerSocket:
    def __init__(self, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.recorder_image = RecorderImage()
        self.agv_control = AgvController(agv_ip=AGV_IP)
        self.arm_control = ArmController(robot_ip=ROBOT_IP)

    def send_dict(self, data: dict):
        """Send a dictionary as JSON."""
        # self.socket.send_json(data)
        byte_data = pickle.dumps(data)
        compressed_data = zlib.compress(byte_data)
        self.socket.send(compressed_data)

    def recv_dict(self) -> dict:
        """Receive a dictionary as JSON."""
        # return self.socket.recv_json()
        compressed_data = self.socket.recv()
        decompressed_data = zlib.decompress(compressed_data)
        data = pickle.loads(decompressed_data)
        return data

    # def send_dict(self, data: dict):
    #     """Send a dictionary as JSON."""
    #     self.socket.send_json(data)

    # def recv_dict(self) -> dict:
    #     """Receive a dictionary as JSON."""
    #     return self.socket.recv_json()

    def close(self):
        """Close the socket"""
        self.socket.close()
        self.context.term()

    def handle_message(self, message):  # message['message'] are include in [paths, pose, claw, get_pose, frame, over]
        message_type = message["type"]
        if message_type == "set_paths":
            try:
                paths = message[message_type]
                # transformed_points = self.agv_control.batch_transform(paths)
                self.agv_control.robot_motion(paths)
                self.send_dict({"response": f"\n\nReceive paths: {paths}, AGV are running over!\n\n"})
            except Exception as e:
                self.send_dict({"Exception response": str(e)})
        elif message_type == "set_pose":
            try:
                [pose, used_time] = message[message_type]
                # self.arm_control.move_pvat(pose, used_time=used_time)
                self.arm_control.move_j(pose)
                self.send_dict({"response": f"\n\nReceive pose: {pose}, ARM are moving over!\n\n"})
            except Exception as e:
                self.send_dict({"Exception response": str(e)})
        
        elif message_type == "set_euler_pose":
            try:
                [euler_pose, used_time] = message[message_type]
                pose = self.arm_control.kinematics_inverse(euler_pose)
                # self.arm_control.move_pvat(pose, used_time=used_time)
                self.arm_control.move_j(pose, v=0.1)
                self.send_dict({"response": f"\n\nReceive euler_pose: {pose}, ARM are moving over!\n\n"})
            except Exception as e:
                self.send_dict({"Exception response": str(e)})
        
        elif message_type == "set_claw":
            try:
                claw = message[message_type]  # [force, amplitude]
                self.arm_control.set_claw(claw[0], claw[1])
                self.send_dict({"response": f"\n\nReceive claw: {pose}, ARM are moving over!\n\n"})
            except Exception as e:
                self.send_dict({"Exception response": str(e)})

        elif message_type == "check_pickup":
            try:
                is_pickup = self.arm_control.check_pickup()
                self.send_dict({"is_pickup": is_pickup,})
            except Exception as e:
                self.send_dict({"Exception response": str(e)})

        elif message_type == "get_pose":
            try:
                pose = self.arm_control.get_actual_flange_pose()
                self.send_dict({"pose": pose})
            except Exception as e:
                self.send_dict({"Exception response": str(e)})

        elif message_type == "get_frame":
            try:
                img_h, img_w = message[message_type]
                recorder_success, depth, color, point, mask, calib, dist_coef = self.recorder_image.get_one_align_frame()
                pose = self.arm_control.get_actual_flange_pose()
                if not recorder_success:
                    self.send_dict({"response": "\n\nERROR: Image Recorder Error Happend!\n\n"})
                else:
                    self.send_dict({
                        "depth": depth.tolist(),
                        "color": color.tolist(),
                        "point": point.tolist(),
                        "mask": mask.tolist(),
                        "calib": calib.tolist(),
                        "pose": pose,
                        "dist_coef": dist_coef.tolist()
                    })
            except Exception as e:
                self.send_dict({"Exception response": str(e)})
        
        elif message_type == "get_see_frames":
            try:
                # move robot to init pose
                self.arm_control.move_init_pose()
                # first frames for over scene
                self.arm_control.running_scene(self.arm_control.see_frames_0_scene, is_wait=True)
                recorder_success, depth, color, point, mask, calib, dist_coef = self.recorder_image.get_one_align_frame()
                if not recorder_success:
                    self.send_dict({"response": "\n\nERROR: Image Recorder Error Happend!\n\n"})
                    return False
                pose = self.arm_control.get_actual_flange_pose()
                depths = [depth]
                colors = [color]
                points = [point]
                masks = [mask]
                calibs = [calib]
                poses = [pose]
                dist_coefs = [dist_coef]
                for i in range(self.arm_control.see_pose_others.shape[0]):
                    # move arm
                    target_pose = self.arm_control.see_pose_others[i].tolist()
                    self.arm_control.move_j(target_pose)
                    time.sleep(1)
                    # recorder image
                    recorder_success, depth, color, point, mask, calib, dist_coef = self.recorder_image.get_one_align_frame()
                    if not recorder_success:
                        self.send_dict({"response": "\n\nERROR: Image Recorder Error Happend!\n\n"})
                        return False
                    # get arm pose
                    pose = self.arm_control.get_actual_flange_pose()
                    depths.append(depth)
                    colors.append(color)
                    points.append(point)
                    masks.append(mask)
                    calibs.append(calib)
                    poses.append(pose)
                    dist_coefs.append(dist_coef)

                depths = np.stack(depths, axis=0)
                colors = np.stack(colors, axis=0)
                points = np.stack(points, axis=0)
                masks = np.stack(masks, axis=0)
                calibs = np.stack(calibs, axis=0)
                dist_coefs = np.stack(dist_coefs, axis=0)
                self.send_dict({
                    "depths": depths.tolist(),
                    "colors": colors.tolist(),
                    "points": points.tolist(),
                    "masks": masks.tolist(),
                    "calibs": calibs.tolist(),
                    "dist_coefs": dist_coefs.tolist(),
                    "poses": poses,
                })
                print("time:", time.time(), "move arm to init pose")
                self.arm_control.move_init_pose()
            except Exception as e:
                self.send_dict({"Exception response": str(e)})
                print("time:", time.time(), "move arm to init pose", str(e))
                self.arm_control.move_init_pose(used_time=5)

        elif message_type == "pickup":
            try:
                pickup_pose = message[message_type]
                self.arm_control.pickup(pickup_pose)
                self.send_dict({"response": f"\n\nPickup Over!\n\n"})
            except Exception as e:
                self.send_dict({"Exception response": str(e)})
        elif message_type == "place":
            try:
                place_pose = message[message_type]
                self.arm_control.place(place_pose)
                self.send_dict({"response": f"\n\nPickup Over!\n\n"})
            except Exception as e:
                self.send_dict({"Exception response": str(e)})
        elif message_type == "close":
            self.send_dict({"response": "\n\nRemote are close!\n\n"})
            self.close()
            self.arm_control.stop_sys()
            self.agv_control.terminate()
            self.recorder_image.pipeline.stop()
            return True
        return False

def start_server(port="9999"):
    server_socket = ZmqServerSocket(port)
    print("time:", time.time(), f"[*] Server is listening on port {port}")
    while True:
        try:
            message = server_socket.recv_dict()
            is_close = server_socket.handle_message(message)
            if is_close:
                print("time:", time.time(), "socket closed!")
                break
        except Exception as e:
            server_socket.send_dict({"Exception response": str(e)})

if __name__ == '__main__':
    start_server()

