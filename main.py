""""
对各个方法模块进行打包
读取.bag文件
对路沿进行实例分割
将分割后的图形映射到点云图中
对点云图中的路沿边缘点进行控制点提取并生成贝塞尔曲线
按键"S"进行保存总点云图和分割点云图
同时引入目标检测模和实例分割模型
在将目标检测模型进行中心点定位
处理实例分割结果，只保留像素数量最多的掩码
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import open3d as o3d  # 用于显示点云
import time
import pandas as pd
from scipy.interpolate import make_interp_spline
from method.method_pre import (project_to_image,load_bag_file,save_point_cloud_to_ply,
                               get_point_cloud_from_mask,handle_frame,get_voxelized_point_cloud_from_mask)
from method.method_algorithm import draw_bezier_curve,extract_and_draw_leftmost_points

# 加载 YOLO 模型并将其移至GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seg_model = YOLO('weights/best1.pt').to(device)  # 用于路沿分割的模型
det_model = YOLO('weights/yolov8n.pt').to(device)  # 用于目标检测的模型

# 启用自动混合精度
scaler = torch.cuda.amp.GradScaler()

def process_bag_file(file_path, seg_model, target_fps=30):
    """
    处理 .bag 文件并实时同步显示 YOLOv8 处理后的图像、原始深度图和生成的点云。
+
032.+0
    Args:
        file_path (str): .bag 文件的路径。
        model (YOLO): 已加载的 YOLO 模型。
        target_fps (int, optional): 目标帧率。默认为 15。
    """
    pipeline, align = load_bag_file(file_path)  # 加载 bag 文件并初始化 pipeline 和 align 对象
    frame_time = 1.0 / target_fps  # 每帧的目标时间间隔

    try:
        device = pipeline.get_active_profile().get_device()  # 获取当前 pipeline 关联的设备
        if device.is_playback():
            playback = device.as_playback()
            playback.set_real_time(False)  # 关闭实时播放模式，确保数据处理不会受限于播放速度

        # 创建 Open3D 可视化器
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(window_name='Point Cloud')

        frame_count = 0  # 用于区分每一帧
        while True:
            start_time = time.time()  # 记录帧开始处理的时间

            # 1. 等待并获取当前帧
            frame_fetch_start = time.time()

            frames = pipeline.wait_for_frames()  # 等待并获取当前帧
            aligned_frames = align.process(frames)  # 对齐深度和颜色帧
            depth_frame = aligned_frames.get_depth_frame()  # 获取对齐后的深度帧
            color_frame = aligned_frames.get_color_frame()  # 获取对齐后的颜色帧
            frame_fetch_end = time.time()
            print(f"视频流与帧匹配耗时 {frame_fetch_end - frame_fetch_start:.4f} seconds")

            # 如果没有有效的深度帧或颜色帧，则跳过
            if not depth_frame or not color_frame:
                print("Skipping frame due to missing depth or color data.")
                continue

            # 3. 转换深度图像和颜色图像为 NumPy 数组
            conversion_start = time.time()
            depth_image = np.asanyarray(depth_frame.get_data())  # 将深度数据转换为 NumPy 数组
            color_image = np.asanyarray(color_frame.get_data())  # 将颜色数据转换为 NumPy 数组
            depth_intri = depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度相机的内参
            depth_image_shape = (depth_frame.get_height(), depth_frame.get_width())
            conversion_end = time.time()
            print(f"帧转数组耗时 {conversion_end - conversion_start:.4f} seconds")

            # 使用 YOLO 模型进行实例分割和目标检测
            yolo_start = time.time()
            with torch.cuda.amp.autocast():  # 使用自动混合精度
                seg_results = seg_model(color_image, save=False)  # 路沿分割
                det_results = det_model(color_image, save=False)  # 目标检测

            # 处理实例分割结果，只保留像素数量最多的掩码
            if seg_results and len(seg_results) > 0:
                for result_idx, result in enumerate(seg_results):
                    if result.masks is not None:
                        if hasattr(result.masks, 'data'):
                            masks = result.masks.data.cpu().numpy()  # 将掩码转换为 NumPy 数组
                            # 找出像素数量最多的掩码
                            max_pixels = 0
                            max_mask_idx = 0
                            for i, mask in enumerate(masks):
                                pixels = np.sum(mask)  # 计算掩码中非零像素的数量
                                if pixels > max_pixels:
                                    max_pixels = pixels
                                    max_mask_idx = i
                            # 只保留像素最多的掩码
                            masks = masks[max_mask_idx:max_mask_idx+1]
                            result.masks.data = torch.from_numpy(masks).to(result.masks.data.device)
                        else:
                            raise TypeError(f"Unsupported mask object. Expected tensor-like, got {type(result.masks)}")

            # 处理目标检测结果
            if det_results and len(det_results) > 0:
                for result in det_results:
                    boxes = result.boxes
                    for box in boxes:
                        # 获取类别
                        cls = int(box.cls[0])
                        class_name = det_model.names[cls]
                        
                        # 只处理指定的6类目标
                        if class_name in ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'bus']:
                            # 获取边界框坐标（左上角和右下角的坐标）
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # 绘制边界框
                            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                            
                            # 在边界框左上角绘制类别名称
                            cv2.putText(color_image, class_name, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                            
                            # 计算中心点坐标
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # 获取中心点的深度信息
                            depth = depth_frame.get_distance(center_x, center_y)
                            
                            # 在图像上绘制中心点
                            cv2.circle(color_image, (center_x, center_y), 2, (0, 255, 0), -1)
                            
                            # 将2D像素坐标转换为3D相机坐标
                            if depth > 0:
                                # 使用rs.rs2_deproject_pixel_to_point将像素坐标和深度值转换为相机坐标系下的3D点
                                point_3d = rs.rs2_deproject_pixel_to_point(depth_intri, [center_x, center_y], depth)
                                
                                # 显示3D位置信息
                                position_text = f"({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f})m"
                                cv2.putText(color_image, position_text, (x1, y1-25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                            else:
                                # 如果深度值无效，显示提示信息
                                cv2.putText(color_image, "深度无效", (center_x + 10, center_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            
            # 显示分割结果
            annotated_frame = seg_results[0].plot()
            cv2.imshow("YOLO Instance Segmentation", annotated_frame)

            yolo_end = time.time()
            print(f"YOLO模型预测耗时 {yolo_end - yolo_start:.4f} seconds")

            mask_process_start = time.time()
            if seg_results and len(seg_results) > 0:
                for result_idx, result in enumerate(seg_results):
                    if result.masks is not None:
                        if hasattr(result.masks, 'data'):
                            masks = result.masks.data.cpu().numpy()  # 将掩码转换为 NumPy 数组
                        else:
                            raise TypeError(f"Unsupported mask object. Expected tensor-like, got {type(result.masks)}")

                        for mask_idx, mask in enumerate(masks):
                            # 根据掩码生成点云和对应的像素坐标
                            point_cloud, pixel_coords = get_point_cloud_from_mask(depth_frame, depth_intri, mask,
                                                                                  depth_image_shape)
                            # # 体素化法三维重构，效果太差
                            # point_cloud=get_voxelized_point_cloud_from_mask(depth_frame, depth_intri, mask,
                            #                                                        depth_image_shape)
                            """
                            保存总体点云和分割后得到的点云
                            """
                            # 检测按键
                            key = cv2.waitKey(1) & 0xFF  # 等待按键
                            if key == ord('s'):  # 按 's' 键保存点云
                                color_image = color_image.reshape(-1, 3)  # 将图像展平以便与点云配对

                                output_maskPointCloud_filename = f"fantou_maskPointCloud_frame{frame_count}_mask_{mask_idx}.ply"
                                save_point_cloud_to_ply(output_maskPointCloud_filename, point_cloud)

                                if not depth_frame:
                                    print("Error: Depth frame is empty.")
                                else:
                                    pointcloud = rs.pointcloud()
                                    full_points = pointcloud.calculate(depth_frame)
                                    vertices = np.asanyarray(full_points.get_vertices()).view(np.float32).reshape(-1, 3)
                                depth_data = np.asanyarray(depth_frame.get_data())

                                # output_fullPointCloud_filename = f"output_fullPointCloud_frame{frame_count}_mask_{mask_idx}.ply"
                                # save_point_cloud_to_ply(output_fullPointCloud_filename, vertices)

                                print(f"Mask point cloud saved as {output_maskPointCloud_filename}")
                                # print(f"Full cloud with color saved as {output_fullPointCloud_filename}")
                            elif key == ord('q'):  # 按 'q' 键退出
                                break

                            mask_process_1 = time.time()
                            print(f"由RGB掩码映射提取点云坐及RGB坐标标耗时 "
                                  f"{mask_process_1 - mask_process_start:.4f} seconds")

                            mask_process_3 = time.time()
                            print(f"显示点云标耗时 "
                                  f"{mask_process_3 - mask_process_1:.4f} seconds")

                            # 构造相机内参矩阵
                            fx = 386.888  # 焦距
                            fy = 385.531  # 焦距
                            cx = 321.010  # 光学中心X
                            cy = 246.484  # 光学中心Y
                            camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                            # 在 RGB 图像上绘制定位点
                            extract_and_draw_leftmost_points(point_cloud, color_image, camera_intrinsics)
                            mask_process_4 = time.time()
                            print(f"RGB图像绘制定位点及贝塞尔曲线耗时 "
                                  f"{mask_process_4 - mask_process_3:.4f} seconds")

                    # 显示 YOLO 处理后的分割图像
                    im_array = result.plot()
                    cv2.imshow('YOLO Instance Segmentation', im_array)
                    # 设置固定尺寸的可视化屏幕
                    # resized_frame = cv2.resize(color_image, (1920, 1080))
                    # cv2.imshow("Video", resized_frame)

                    cv2.imshow('Anchor point', color_image)

            else:
                print("No objects detected in this frame.")

            display_start = time.time()
            # 显示原始深度图像
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.07), cv2.COLORMAP_JET)
            cv2.imshow('Original Depth Image', depth_colormap)

            # 显示在 RGB 图像上绘制了点的图像
            cv2.imshow('RGB Image with Points', color_image)
            display_end = time.time()
            print(f"显示图像耗时 {display_end - display_start:.4f} seconds")
            frame_count += 1  # 更新帧计数器

            # 计算并打印每步处理时间和每帧处理时间
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Frame {frame_count} 总耗时 {processing_time:.4f} seconds")

            # 计算并打印实际的FPS
            actual_fps = 1.0 / processing_time
            print(f"Actual FPS: {actual_fps:.2f}")

            # 限制帧率`
            sleep_time = frame_time - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)  # 只有当处理时间小于q目标时间时才休

            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        # vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    bag_file_path = "output_chair1.bag"

    # Loop indefinitely to process the bag file continuously
    while True:
        # 每次循环重新初始化 pipeline 和 align 对象
        pipeline, align = load_bag_file(bag_file_path)
        try:
            process_bag_file(bag_file_path, seg_model)
        finally:
            pipeline.stop()  # 确保在每次循环结束时停止 pipeline

        # 添加一个小的延迟，避免过于频繁地重新初始化
        time.sleep(10)