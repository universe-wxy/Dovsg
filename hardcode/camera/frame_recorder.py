import pyrealsense2 as rs
import numpy as np
import cv2
import copy

# The actual image size required to ensure that 
# the depth and color closer to the camera are not captured
real_height = 600  # remove bottom 200px to remove robot arm in image bottom
real_width = 1200

class RecorderImage():
    def __init__(self, serial_number="309122301398", 
                 WH=[1280, 720], FPS=30, depth_threshold=[0, 3]):
        
        self.WH = WH
        self.serial_number = serial_number
        self.FPS = FPS
        self.depth_threshold = depth_threshold
        self.config = rs.config()
        # Specify the wrist camera serial number
        self.config.enable_device(self.serial_number)
        self.pipeline = rs.pipeline()
        self.config.enable_stream(rs.stream.depth, self.WH[0], self.WH[1], rs.format.z16, self.FPS)
        self.config.enable_stream(rs.stream.color, self.WH[0], self.WH[1], rs.format.bgr8, self.FPS)
        profile = self.pipeline.start(self.config)
        # Skip 15 first frames to give the Auto-Exposure time to adjust
        for x in range(15):
            self.pipeline.wait_for_frames()
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsic = color_stream.as_video_stream_profile().get_intrinsics()

        self.intrinsic_matrix, self.dist_coef = self._get_readable_intrinsic()
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.depth_scale)
        self.clipping_distance = 1 / self.depth_scale
        print(self.clipping_distance)
        # Initialize depth process
        self._init_depth_process()

    def _init_depth_process(self):
        # Initialize the processing steps
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.spatial.set_option(rs.option.filter_smooth_delta, 1)
        self.spatial.set_option(rs.option.holes_fill, 1)
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.temporal.set_option(rs.option.filter_smooth_delta, 1)
        # Initialize the alignment to make the depth data aligned to the rgb camera coordinate
        self.align = rs.align(rs.stream.color)

    def _get_readable_intrinsic(self):
        intrinsic_matrix = np.array(
            [
                [self.intrinsic.fx, 0, self.intrinsic.ppx],
                [0, self.intrinsic.fy, self.intrinsic.ppy],
                [0, 0, 1],
            ], dtype=np.float32
        )
        dist_coef = np.array(self.intrinsic.coeffs, dtype=np.float32)
        return intrinsic_matrix, dist_coef


    def project_point_to_pixel(self, points):
        # The points here should be in the camera coordinate, n*3
        points = np.array(points)
        pixels = []
        # # Use the realsense projeciton, however, it's slow for the loop; this can give nan for invalid points
        # for i in range(len(points)):
        #     pixels.append(rs.rs2_project_point_to_pixel(self.intrinsic, points))
        # pixels = np.array(pixels)

        # Use the opencv projection
        # The width and height are inversed here
        pixels = cv2.projectPoints(
            points,
            np.zeros(3),
            np.zeros(3),
            self.intrinsic_matrix,
            self.dist_coef,
        )[0][:, 0, :]

        return pixels[:, ::-1]

    def deproject_pixel_to_point(self, pixel_depth):
        # pixel_depth contains [i, j, depth[i, j]]
        points = []
        for i in range(len(pixel_depth)):
            # The width and height are inversed here
            points.append(
                rs.rs2_deproject_pixel_to_point(
                    self.intrinsic,
                    [pixel_depth[i, 1], pixel_depth[i, 0]],
                    pixel_depth[i, 2],
                )
            )
        return np.array(points)

    def _process_depth(self, depth_frame):
        # Depth process
        filtered_depth = self.depth_to_disparity.process(depth_frame)
        filtered_depth = self.spatial.process(filtered_depth)
        filtered_depth = self.temporal.process(filtered_depth)
        filtered_depth = self.disparity_to_depth.process(filtered_depth)
        return filtered_depth

    def get_observations(self):
        width, height = self.WH
        depth_frame = None
        for x in range(5):  # each frame past five frame
            self.pipeline.wait_for_frames()
        while not depth_frame:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

        # Depth process
        filtered_depth = self._process_depth(depth_frame)
        # Calculate the pointcloud
        pointcloud = rs.pointcloud()
        pointcloud.map_to(color_frame)
        pointcloud = pointcloud.calculate(filtered_depth)
        # Get the 3D points and colors
        points = (
            np.asanyarray(pointcloud.get_vertices())
            .view(np.float32)
            .reshape([height, width, 3])
        )
        # Convert the colors from BGR to RGB
        # colors = (np.asanyarray(color_frame.get_data()) / 255.0)[:, :, ::-1]
        # # Get the depth image, make the depth in meters
        # depths = np.asanyarray(filtered_depth.get_data()) / self.clipping_distance
        colors = np.asanyarray(color_frame.get_data(), dtype=np.float32)
        # Get the depth image, make the depth in meters
        depths = np.asanyarray(filtered_depth.get_data(), dtype=np.float32)
        # Get the mask of the valid depth in the depth threshold
        mask = np.logical_and(
            (depths > self.depth_threshold[0] * self.clipping_distance), 
            (depths < self.depth_threshold[1] * self.clipping_distance)
        )
        return points, colors, depths, mask


    def get_one_align_frame(self, retry=10):
        for i in range(retry):
            try:
                point, color, depth, mask = self.get_observations()
                point = point[:600, :]
                color = color[:600, :]
                depth = depth[:600, :]
                mask = mask[:600, :]

                print("One Frame Depth and rgb Recieved.")
                return True, depth, color, point, mask, self.intrinsic_matrix, self.dist_coef
            except Exception as e:
                print(f"Dont get valid depth and rgb, try again. {e}")
        return False, None, None, None, None, None


if __name__ == "__main__":
    record = RecorderImage()
