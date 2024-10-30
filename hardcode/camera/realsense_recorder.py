import pyrealsense2 as rs
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from hardcode.robotarm.arm_control import ArmController
import os
import shutil
import cv2
import time
import threading
from IPython import embed
import lebai_sdk
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from utils import LOOP_SCENE_ID, RECOVERY_SCENE_ID, SEE_SCENE_ID
from utils import T_cam_in_end, RECORDER_DIR
from utils import DEPTH_MIN, DEPTH_MAX

class RecorderImage():
    def __init__(self, recorder_dir=None, robot=None, color_width=640, color_height=480, 
                 depth_width=640, depth_height=480, FPS=30, need_width=None, need_height=None):
        self.robot = robot
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.clipping_distance_in_meters_max = DEPTH_MAX
        self.clipping_distance_in_meters_min = DEPTH_MIN
        self.frame_index = 0
        self.color_width = color_width
        self.color_height = color_height
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.FPS = FPS
        self.ori_intrinsics = {}

        self.need_width = need_width
        self.need_height = need_height

        if recorder_dir is not None:
            self.record_flag = True
            # self.color_images = {}
            # self.depth_images = {}
            self.recorder_dir = recorder_dir
            self.transforms = {}

            if os.path.isdir(self.recorder_dir):
                if input("Overwrite file y/n?: ") == "y":
                    shutil.rmtree(self.recorder_dir)
                else:
                    return
            os.makedirs(self.recorder_dir / "depth", exist_ok=True)
            os.makedirs(self.recorder_dir / "color", exist_ok=True)

    def set_config(self):
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        # device_product_line = str(device.get_info(rs.camera_info.product_line))

        device_serial_number = device.get_info(rs.camera_info.serial_number)
        print(device_serial_number)
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.config.enable_stream(rs.stream.depth, self.color_width, self.color_height, rs.format.z16, self.FPS)
        self.config.enable_stream(rs.stream.color, self.depth_width, self.depth_height, rs.format.bgr8, self.FPS)

        # Start streaming
        profile = self.pipeline.start(self.config)

        self.color_stream_profile = profile.get_stream(rs.stream.color)
        
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        self.clipping_distance = 1 / self.depth_scale
        print(self.clipping_distance)
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        ori_intrinsics = self.color_stream_profile.as_video_stream_profile().get_intrinsics()
        self.ori_intrinsics = {
            "width": ori_intrinsics.width,
            "height": ori_intrinsics.height,
            "fx": ori_intrinsics.fx,
            "fy": ori_intrinsics.fy,
            "ppx": ori_intrinsics.ppx,
            "ppy": ori_intrinsics.ppy,
            "scale": self.depth_scale
        }
        print(self.ori_intrinsics)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_intrinsics(self, new_width=None, new_height=None):
        ori_intrinsics = self.color_stream_profile.as_video_stream_profile().get_intrinsics()
        if new_width and new_height:
            intrinsics = {
                # "coeffs": intrinsics.coeffs,
                "width": self.need_width,
                "height": self.need_height,
                # "model": intrinsics.model,
                "fx": ori_intrinsics.fx * new_width / self.color_width,
                "fy": ori_intrinsics.fy * new_height / self.color_height,
                "ppx": ori_intrinsics.ppx * new_width / self.color_width,
                "ppy": ori_intrinsics.ppy * new_height / self.color_height,
                "scale": self.depth_scale
            }
        else:
           intrinsics = {
                # "coeffs": intrinsics.coeffs,
                "width": ori_intrinsics.width,
                "height": ori_intrinsics.height,
                # "model": intrinsics.model,
                "fx": ori_intrinsics.fx,
                "fy": ori_intrinsics.fy,
                "ppx": ori_intrinsics.ppx,
                "ppy": ori_intrinsics.ppy,
                "scale": self.depth_scale
            }
        return intrinsics

    def get_align_frame(self, frame_index):
        # Streaming loop
        try:
            # Get frameset of color and depth
            frames = self.pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                return False
            
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())  # BGR

            if self.need_width and self.need_height:
                depth_image = cv2.resize(depth_image, (self.need_width, self.need_height), interpolation=cv2.INTER_NEAREST)
                color_image = cv2.resize(color_image, (self.need_width, self.need_height), interpolation=cv2.INTER_LINEAR)

            flange_pose = self.robot.get_actual_flange_pose()
            isinstance(flange_pose, dict)
            
            depth_image_path = self.recorder_dir / "depth" / f"{frame_index}.png"
            color_image_path = self.recorder_dir / "color" / f"{frame_index}.png"
            cv2.imwrite(depth_image_path, depth_image)
            cv2.imwrite(color_image_path, color_image)
            self.transforms[f"{frame_index}"] = flange_pose
            
            # self.depth_images[f"{frame_index}.png"] = depth_image
            # self.color_images[f"{frame_index}.png"] = color_image
        except:
            return False
        return True

    def get_one_align_frame(self, depth_path, color_path, retry=10):
        self.robot.running_scene(SEE_SCENE_ID, is_wait=True)
        for i in range(retry):
            try:
                # Get frameset of color and depth
                frames = self.pipeline.wait_for_frames()
                # Align the depth frame to color frame
                aligned_frames = self.align.process(frames)
                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()
                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue
                
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())  # BGR
                
                # original_height, original_width = self.color_height, self.color_width
                new_height, new_width = 240, 320

                depth_image_resized = cv2.resize(depth_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                color_image_resized = cv2.resize(color_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                self.intrinsics_resized = self.get_intrinsics(new_width=new_width, new_height=new_height)
                print(f"intrinsics: {self.get_intrinsics()}")
                print(f"intrinsics_resized: {self.intrinsics_resized}")

                cv2.imwrite(depth_path, depth_image_resized)
                cv2.imwrite(color_path, color_image_resized)
 
                print(depth_path)
                print(color_path)
                print("One Frame Depth and Color Recieved.")
                break
            except:
                print("Dont get valid depth and color, try again.")

    def set_metadata(self):
        if self.need_width and self.need_height:
            intrinsics = self.get_intrinsics(new_width=self.need_width, new_height=self.need_height)
        else:
            intrinsics = self.get_intrinsics()
        return {
                    "w": self.need_width if self.need_width else self.color_width,
                    "h": self.need_height if self.need_height else self.color_height, 
                    "dw": self.need_width if self.need_width else self.depth_width, 
                    "dh": self.need_height if self.need_height else self.depth_height,
                    "fps": self.FPS,
                    "K": [intrinsics["fx"], 0.0, intrinsics["ppx"], 0.0, intrinsics["fy"], intrinsics["ppy"], 0.0, 0.0, 1.0],
                    "depth_scale": self.depth_scale,
                    "min_depth": 0.2,
                    "max_depth": 2.5,
                    "initPose": [0, 0, 0, 1, 0, 0, 0],  # TODO
                    "cameraType": 1, 
                    "T_cam_in_end": T_cam_in_end.tolist()
                }

    def start_record(self):
        while self.record_flag:
            flag = self.get_align_frame(self.frame_index)
            # time.sleep(0.001)
            time.sleep(0.1)
            # print(self.frame_index)
            if flag: self.frame_index += 1
    
    def stop_record(self):
        self.record_flag = False
        self.pipeline.stop()
    
    def depth_to_color_vis(self):
        self.depth_dir = self.recorder_dir / "depth"
        self.color_dir = self.recorder_dir / "color"
        self.depth_color_dir = self.recorder_dir / "depth_color"
        self.depth_color_dir.mkdir(parents=True, exist_ok=True)
        
        depth_filepaths = [self.color_dir / f for f in sorted(os.listdir(self.color_dir), key=lambda x: int(x.split(".")[0]))]
        color_filepaths = [self.color_dir / f for f in sorted(os.listdir(self.color_dir), key=lambda x: int(x.split(".")[0]))]
        

        assert len(depth_filepaths) == len(color_filepaths)

        # for cnt in tqdm(range(len(depth_filepaths)), desc="Save depth color Images."):
        for cnt in tqdm(range(10), desc="Save depth color Images."):
            color_path = color_filepaths[cnt]
            depth_path = depth_filepaths[cnt]

            color = cv2.imread(color_path, cv2.IMREAD_COLOR)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

            alpha = 0.8
            overlay_image = cv2.addWeighted(color, alpha, depth_colored, 1 - alpha, 0)

            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            plt.title('RGB Image')
            plt.imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))

            plt.subplot(1, 3, 2)
            plt.title('Depth Image')
            plt.imshow(depth_colored, cmap='jet')  # 确保显示的颜色映射与上面一致

            plt.subplot(1, 3, 3)
            plt.title('Overlay Image')
            plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))

            plt.savefig(self.depth_color_dir / (color_path.stem + ".png"), bbox_inches="tight")
            plt.close()

def record():
    if input("Do you want to record data? [y/n]: ") == "n":
        return 
    
    arm_control = ArmController()
    arm_control.start_sys()
    arm_control.set_claw(10, 100)  # set claw for record sense without finger
    
    arm_control.running_scene(RECOVERY_SCENE_ID, is_wait=True)  # ROBOT
    recorder_dir = RECORDER_DIR / "r3d_py"
    imagerecorder = RecorderImage(recorder_dir=recorder_dir, robot=arm_control)
    imagerecorder.set_config()

    input("\033[32mPress any key to Start.\033[0m")

    record_thread = threading.Thread(target=imagerecorder.start_record)
    record_thread.start()

    task_id = arm_control.running_scene(LOOP_SCENE_ID)  # ROBOT
    input("\033[31mRecording started. Press any key to stop.\033[0m")
    arm_control.cancel_task(task_id)  # ROBOT
    imagerecorder.stop_record()

    with open(recorder_dir / "poses.json", "w") as f:
        json.dump(imagerecorder.transforms, f)
    record_thread.join()

    with open(recorder_dir / "intrinsics.json", "w") as f:
        json.dump(imagerecorder.ori_intrinsics, f, indent=4)
    
    metadata = imagerecorder.set_metadata()
    with open(recorder_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("====> Recover robot arm start position.")
    arm_control.running_scene(RECOVERY_SCENE_ID)  # ROBOT
    imagerecorder.depth_to_color_vis()
    print(f"All Images are save in {imagerecorder.recorder_dir}: depth / color, length is {len(os.listdir(imagerecorder.recorder_dir / 'depth'))}")
    del imagerecorder

if __name__ == "__main__":
    record()
