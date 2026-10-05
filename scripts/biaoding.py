import pyrealsense2 as rs
import numpy as np

"""
相机内参测量
"""
def get_camera_intrinsics():
    # 配置 RealSense 流
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置 RGB 流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)    # 配置深度流

    # 启动流
    pipeline.start(config)

    # 等待一帧
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # 获取内参
    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

    # 打印内参
    print("Camera Intrinsics:")
    print(f"  Width: {intrinsics.width}")
    print(f"  Height: {intrinsics.height}")
    print(f"  FX: {intrinsics.fx}")
    print(f"  FY: {intrinsics.fy}")
    print(f"  CX: {intrinsics.ppx}")
    print(f"  CY: {intrinsics.ppy}")
    print(f"  Distortion Mo del: {intrinsics.model}")
    print(f"  Distortion Coefficients: {intrinsics.coeffs}")

    # 停止流
    pipeline.stop()

if __name__ == "__main__":
    get_camera_intrinsics()
