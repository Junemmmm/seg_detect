import pyrealsense2 as rs
import numpy as np
import cv2


def record_and_visualize_bag(output_file):
    # 创建 pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置要输出的 .bag 文件
    config.enable_record_to_file(output_file)

    # 启用彩色、深度、加速度计和陀螺仪数据流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)

    # 开始捕获
    pipeline.start(config)

    try:
        print("Recording and visualizing... Press 'q' to stop.")
        while True:
            # 获取帧并处理
            frames = pipeline.wait_for_frames()

            # 获取彩色图像和深度图像
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 将彩色图像转换为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())

            # 将深度图像转换为 numpy 数组
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 合并彩色图和深度图进行可视化
            images = np.hstack((color_image, depth_colormap))

            # 显示图片
            cv2.imshow('RealSense', images)

            # 打印 IMU 数据
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)

            if accel_frame and gyro_frame:
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                print(f"Accel: {accel_data}, Gyro: {gyro_data}")

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 停止 pipeline
        pipeline.stop()
        cv2.destroyAllWindows()


# 示例用法
record_and_visualize_bag("../imu_output_chair1.bag")
