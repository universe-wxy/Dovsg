import zmq
import zlib
from robot_control import RobotControl
import time
import pickle
import sys

"""
# run this scrip, you should running
source ~/agilex_ws/devel/setup.bash
rosrun ranger_bringup bringup_can2usb.bash
roslaunch ranger_bringup ranger_mini_v2.launch

kill port: sudo lsof -i:9999
"""

class ZmqServerSocket:
    def __init__(self, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.robotcontrol = RobotControl()

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

    def close(self):
        """Close the socket"""
        self.socket.close()
        self.context.term()

    def handle_message(self, message):
        message_type = message["type"]
        if message_type == "motion_moving":
            try:
                paths = message[message_type]
                # transformed_points = self.agv_control.batch_transform(paths)
                self.robotcontrol.motion_moving(paths)
                self.send_dict({"response": f"\n\nReceive paths: {paths}, AGV are running over!\n\n"})
            except Exception as e:
                self.send_dict({"Exception response": str(e)})
        elif message_type == "arm_run_action":
            try:
                action_dict = message[message_type]
                action_code = action_dict["action_code"]
                action_parameters = action_dict["action_parameters"]
                speed = action_dict["speed"]
                self.robotcontrol.arm_run_action(
                    action_code=action_code,
                    action_parameters=action_parameters,
                    speed=speed
                )
                self.send_dict({"response": f"\n\nReceive pose: {action_dict}, arm are moving over!\n\n"})
            except Exception as e:
                self.send_dict({"Exception response": str(e)})

        elif message_type == "pick_up":
            try:
                action_dict = message[message_type]
                action_parameters = action_dict["action_parameters"]
                speed = action_dict["speed"]
                pick_back = False
                if "pick_back" in action_dict.keys():
                    pick_back = action_dict["pick_back"]

                self.robotcontrol.pick_up(
                    action_parameters=action_parameters,
                    speed=speed,
                    pick_back=pick_back
                )
                self.send_dict({"response": f"\n\nReceive pose: {action_dict}, arm are moving over!\n\n"})
            except Exception as e:
                self.send_dict({"Exception response": str(e)})

        elif message_type == "get_back_observation":
            try:
                back_observation = self.robotcontrol.get_back_observation()
                self.send_dict(back_observation)
            except Exception as e:
                self.send_dict({"Exception response": str(e)})

        elif message_type == "place":
            try:
                action_dict = message[message_type]
                action_parameters = action_dict["action_parameters"]
                speed = action_dict["speed"]
                
                self.robotcontrol.place(
                    action_parameters=action_parameters,
                    speed=speed
                )
                self.send_dict({"response": f"\n\nReceive pose: {action_dict}, arm are moving over!\n\n"})
            except Exception as e:
                self.send_dict({"Exception response": str(e)})

        elif message_type == "get_observations":
            try:
                params_dict = message[message_type]
                print(params_dict)
                just_wrist = params_dict["just_wrist"]
                observations = self.robotcontrol.get_observations(just_wrist=just_wrist)
                self.send_dict(observations)
                self.robotcontrol.arm.reset() # arm to init pose
            except Exception as e:
                self.send_dict({"Exception response": str(e)})

        elif message_type == "close":
            self.send_dict({"response": "\n\nRemote are close!\n\n"})
            self.close()
            self.robotcontrol.arm.reset()
            self.robotcontrol.base.terminate()
            # self.robotcontrol.camera_top.pipeline.stop()
            self.robotcontrol.camera_wrist.pipeline.stop()
            return True
        return False

def start_server(port="9999"):
    server_socket = ZmqServerSocket(port)
    print(f"[*] Server is listening on port {port}")
    while True:
        try:
            message = server_socket.recv_dict()
            is_close = server_socket.handle_message(message)
            if is_close:
                print("socket closed!")
                server_socket.close()
                break
        except KeyboardInterrupt:
            server_socket.handle_message("close")
            # server_socket.close()
            sys.exit(0)
            break
        except Exception as e:
            # server_socket.send_dict({"Exception response": str(e)})
            server_socket.handle_message("close")
            # server_socket.close()
            sys.exit(0)
            break


if __name__ == '__main__':
    start_server()

