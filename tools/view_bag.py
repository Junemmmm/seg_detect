"""
查看.bag文件保存的视频
"""

import pyrealsense2 as rs

# 创建RealSense管道
pipeline = rs.pipeline()

# 创建配置对象
config = rs.config()

# 让管道从.bag文件中读取数据
config.enable_device_from_file('../new12.bag')

# 开始播放.bag文件
pipeline.start(config)

try:
    while True:
        # 等待下一帧
        frames = pipeline.wait_for_frames()

        # 获取深度和颜色帧
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # 检查帧是否有效
        if not depth_frame or not color_frame:
            continue

        # 获取流配置的帧率和分辨率
        depth_profile = depth_frame.get_profile()
        color_profile = color_frame.get_profile()

        # 获取深度帧的帧率和分辨率
        depth_fps = depth_profile.fps()
        depth_resolution = (depth_profile.as_video_stream_profile().width(),
                            depth_profile.as_video_stream_profile().height())

        # 获取颜色帧的帧率和分辨率
        color_fps = color_profile.fps()
        color_resolution = (color_profile.as_video_stream_profile().width(),
                            color_profile.as_video_stream_profile().height())

        print(f"Depth Frame: {depth_fps} FPS, Resolution: {depth_resolution}")
        print(f"Color Frame: {color_fps} FPS, Resolution: {color_resolution}")

        # 如果只想打印一次，去掉下面的break
        break

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    # 停止管道
    pipeline.stop()
