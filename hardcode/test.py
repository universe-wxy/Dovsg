import zmq
import numpy as np
import pickle
import zlib
import open3d as o3d


class ZmqClientSocket:
    def __init__(self, ip="192.168.1.54", port="9999"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{ip}:{port}")

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
        """Close the socket."""
        self.socket.close()
        self.context.term()


class ZmqSocket:
    def __init__(self, ip="192.168.1.54", port="9999"):
        self.client = ZmqClientSocket(ip, port)
    
    def send_info(self, info, type):  # type are include in [paths, pose, frame, over]
        try:
            message = {
                "type": f"{type}",
                f"{type}": info
            }
            self.client.send_dict(message)
        except Exception as e:
            print(f"Error: {e}")

    def received(self):
        recv_info = self.client.recv_dict()
        return recv_info
    
    def close(self):
        self.client.close()

if __name__ == '__main__':
    zmqsocket = ZmqSocket(ip="192.168.1.54", port="9999")
    try:
        zmqsocket.send_info(info={"just_wrist": True}, type="get_observations")
        observations = zmqsocket.received()

        observations_new = {}
        if observations["top"] is not None:
            observations_new[0] = observations["top"]
        start = len(observations_new)
        for i in range(len(observations["wrist"])):
            observations_new[start + i] = observations["wrist"][i]
        if True:
            pcds = []
            for name, obs in observations_new.items():
                # Get the rgb, point cloud, and the camera pose
                color = obs["rgb"]
                point = obs["point"]
                mask = obs["mask"]
                c2b = obs["c2b"]

                point_new = point @ c2b[:3, :3].T + c2b[:3, 3]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_new[mask])
                pcd.colors = o3d.utility.Vector3dVector(color[mask])
                pcds.append(pcd)
            o3d.visualization.draw_geometries(pcds)
    finally:
        zmqsocket.send_info(info="", type="close")
        print(zmqsocket.received())
        zmqsocket.close()
