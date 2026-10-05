import pyrealsense2 as rs
import numpy as np
import cv2
import time

# 设置管道
pipeline = rs.pipeline()
config = rs.config()

# 配置深度流和 RGB 流，设置分辨率为 640x480，帧率为 30
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 设置保存 .bag 文件的路径
bag_file = "../new16.bag"

config.enable_record_to_file(bag_file)

# 开启管道
pipeline.start(config)

try:
    print("Recording .bag file and showing video feed, press 'q' to stop...")
    start_time = time.time()

    while True:
        # 获取帧数据
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("帧捕获失败，跳过当前帧...")
            continue

        # 转换深度帧和RGB帧为NumPy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 将深度数据转换为可视化图像 (灰度图)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 将深度图像和RGB图像水平拼接在一起
        images = np.hstack((color_image, depth_colormap))

        # 使用 OpenCV 显示图像
        cv2.imshow('RealSense RGB and Depth', images)

        # 打印录制时间（每隔5秒更新一次）
        elapsed_time = time.time() - start_time
        if int(elapsed_time) % 5 == 0:
            print(f"录制中... 已运行 {int(elapsed_time)} 秒")

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    # 关闭并停止管道
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"录制完成，文件已保存为 {bag_file}")
