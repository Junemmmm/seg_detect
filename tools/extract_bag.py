import pyrealsense2 as rs
import numpy as np
import cv2


def read_bag_file(bag_file):
    # 创建一个播放管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置从 .bag 文件中播放
    config.enable_device_from_file(bag_file, repeat_playback=True)

    # 开始播放
    pipeline.start(config)

    try:
        while True:
            # 获取帧
            frames = pipeline.wait_for_frames()

            # 获取 IMU 数据
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)

            if accel_frame and gyro_frame:
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                print(f"Accel: {accel_data}, Gyro: {gyro_data}")

            # 获取彩色图像和深度图像
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame and depth_frame:
                # 转换为 numpy 数组
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # 显示图像
                cv2.imshow('Color Image', color_image)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow('Depth Image', depth_colormap)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    read_bag_file("../output_chair9.bag")
