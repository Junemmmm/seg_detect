"""
读取D455相机的数据流
对路沿进行实例分割
将分割后的图形映射到点云图中
对点云图中的路沿边缘点进行控制点提取并生成贝塞尔曲线
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
from sklearn.linear_model import LinearRegression

# 加载 YOLO 模型
model = YOLO('weights/best.pt')

"""
根据相机内参将三维点云点映射到RGB二维图像下
"""
def project_to_image(point_cloud, camera_intrinsics):
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # 将点云中的 X, Y, Z 取出
    X = point_cloud[:, 0]
    Y = point_cloud[:, 1]
    Z = point_cloud[:, 2]

    # 投影到像素坐标
    u = (fx * (X / Z) + cx).astype(int)
    v = (fy * (Y / Z) + cy).astype(int)
    return u, v

"""
创建一个 RealSense 的摄像头数据流（pipeline）和配置（config）。
Args:
     None
Returns:
    pipeline用于处理数据流、glign用于在后续处理中对齐深度和颜色图像数据
"""
def load_realsense_camera():
    """
    创建一个 RealSense 数据流（pipeline）和配置（config）。
    返回 pipeline 和 align 对象。
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置要启用的流类型，例如颜色和深度流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB 图像
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 深度图像

    pipeline.start(config)

    align_to = rs.stream.color  # 设置对齐的目标流类型为颜色图像
    align = rs.align(align_to)  # 创建一个对齐对象
    return pipeline, align

"""
将点云数据保存为 PLY 文件。
Args:
    filename (str): 输出 PLY 文件的路径。
    point_cloud (np.ndarray): 点云数据，形状为 (N, 3)。
"""
def save_point_cloud_to_ply(filename, point_cloud):

    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(point_cloud)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in point_cloud:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

"""
从RGB掩码中生成点云
Args: 
    深度帧、深度相机内、掩码、深度图像shape
Returns: 
    Numopy数组格式的 点云坐标、像素坐标
"""
def get_point_cloud_from_mask(depth_frame, depth_intri, mask, depth_image_shape):
    # 检查掩码mask是否为 PyTorch Tensor格式，如果是则转换为 NumPy 数组
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()  # 将 Tensor 转换为 NumPy 数组格式（只能对tensor格式数据进行转numpy格式操作）
    elif isinstance(mask, np.ndarray):
        mask_np = mask  # 如果掩码已经是 NumPy 数组，直接使用
    else:
        raise TypeError("Unsupported mask type. Expected torch.Tensor or numpy.ndarray.")
    # 判断 mask_np 是否是二维数组
    if len(mask_np.shape) != 2:
        raise ValueError(f"Expected 2D mask, but got shape {mask_np.shape}")

    # 调整掩码到深度图像尺寸
    if mask_np.shape != depth_image_shape:
        # OPENCV是 w x h ，而depth_image_shape 是 h x w（切片操作 逆序）统一到 w x h 再使用插值算法resize
        mask_np = cv2.resize(mask_np, depth_image_shape[::-1])

    # 找到掩码中为 True 或非零的像素点坐标索引，这些索引对应目标物体所在的像素点
    mask_indices = np.argwhere(mask_np)
    points = []  # 存储三维点坐标
    pixel_coords = []  # 存储像素坐标

    for idx in mask_indices:
        uy, ux = idx  # 图像中的 y, x 坐标
        dis = depth_frame.get_distance(ux, uy)  # 获取点的深度值
        # 当拥有深度信息时才会进行记录
        if dis > 0:
            # 校正图像畸变并将像素坐标和深度值转化为相机坐标系下的 (x, y, z) 三维点
            point = rs.rs2_deproject_pixel_to_point(depth_intri, [ux, uy], dis)
            points.append(point)
            pixel_coords.append((ux, uy))  # 保存像素坐标

    return np.array(points), np.array(pixel_coords)

"""
根据控制点绘制贝塞尔曲线
Args: 
    控制点像素下坐标、RGB图像
Returns: 
    绘制好贝塞尔曲线的RGB图像
"""
def draw_bezier_curve(pixel_coords, color_image):
    # 当像素点数不足时返回原图像
    if len(pixel_coords) < 2:
        print("Not enough points to draw a curve.")
        return color_image  # 返回原图像
    # 提取 X 和 Y 坐标
    x_points = pixel_coords[:, 0]
    y_points = pixel_coords[:, 1]
    # 当像素x坐标不足时返回原图像
    if len(x_points) < 2:
        print("Not enough points to draw a curve.")
        return color_image

    # 打印调试信息，检查点的数量和内容
    print(f"x_points: {x_points}, y_points: {y_points}, len: {len(x_points)}")

    # 贝塞尔曲线插值
    num_points = 100    # 定义生成的曲线上的插值点数
    t = np.linspace(0, 1, len(x_points))
    try:
        spl_x = make_interp_spline(t, x_points, k=2 if len(x_points) == 2 else 3)
        spl_y = make_interp_spline(t, y_points, k=2 if len(y_points) == 2 else 3)
    except ValueError as e:
        print(f"Error creating spline: {e}")
        return color_image

    # 生成插值点
    t_new = np.linspace(0, 1, num_points)
    x_smooth = spl_x(t_new)
    y_smooth = spl_y(t_new)

    # 绘制贝塞尔曲线
    for i in range(len(x_smooth) - 1):
        cv2.line(color_image, (int(x_smooth[i]), int(y_smooth[i])),
                 (int(x_smooth[i+1]), int(y_smooth[i+1])), (0, 255, 0), 2)  # 绘制绿色轨迹线

    return color_image

"""
    从点云中按照区间分段提取X轴绝对值最小的点并返回它们的坐标(即)。
    Args:
        输入的点云数据(形状为 (N, 3)), RGB图像， 相机的内参矩阵
    Returns:
        图像上绘制每个区段中X轴绝对值最小的点及坐标信息，以及根据边沿点生成的贝塞尔曲线
"""
def extract_and_draw_leftmost_points(point_cloud, color_image, camera_intrinsics):
    # 检查 point_cloud 是否为空
    if len(point_cloud) == 0:
        print("Warning: point_cloud is empty.")
        return color_image  # 如果点云为空，直接返回原图像

    # 转换为 DataFrame 处理
    cloud_df = pd.DataFrame(point_cloud, columns=['x', 'y', 'z'])

    '''
        按照深度值z进行分段，生成6个区间并选取X轴绝对值最小的点作为控制点
    '''
    depth_bins = np.linspace(cloud_df['z'].min(), cloud_df['z'].max(), 7)  # 生成6个区间
    cloud_df['depth_bin'] = pd.cut(cloud_df['z'], bins=depth_bins)

    selected_points = []
    for bin in cloud_df['depth_bin'].unique():
        bin_points = cloud_df[cloud_df['depth_bin'] == bin]  # 提取当前深度段的点
        if not bin_points.empty:
            # 获取X绝对值最小的点
            rightmost_point = bin_points.loc[bin_points['x'].abs().idxmin()]
            selected_points.append([rightmost_point['x'], rightmost_point['y'], rightmost_point['z']])

    if len(selected_points) == 0:
        print("Warning: No points were selected.")
        return color_image  # 没有选中任何点时返回原图像

    selected_points = np.array(selected_points)

    u_1, v_1 = project_to_image(selected_points, camera_intrinsics)
    '''
        按照深度值z进行分段，生成12个区间并选取X轴绝对值最小的点作为贝塞尔曲线绘制点
    '''
    # 按照深度值z进行分段，并选取X轴绝对值最小的点作为控制点
    depth_curb = np.linspace(cloud_df['z'].min(), cloud_df['z'].max(), 13)  # 生成12个区间
    cloud_df['depth_curb'] = pd.cut(cloud_df['z'], bins=depth_curb)

    depth_curbPoints = []
    for bin in cloud_df['depth_curb'].unique():
        bin_points = cloud_df[cloud_df['depth_curb'] == bin]  # 提取当前深度段的点
        if not bin_points.empty:
            # 获取X绝对值最小的点
            rightmost_point = bin_points.loc[bin_points['x'].abs().idxmin()]
            depth_curbPoints.append([rightmost_point['x'], rightmost_point['y'], rightmost_point['z']])

    depth_curbPoints = np.array(depth_curbPoints)

    # 使用相机内参矩阵将3D点投影到2维图像上
    u, v = project_to_image(depth_curbPoints, camera_intrinsics)

    # 使用投影后的像素坐标绘制贝塞尔曲线
    pixel_coords = np.column_stack((u, v))  # 将u, v坐标合并为 (u, v) 对

    # 绘制贝塞尔曲线
    color_image = draw_bezier_curve(pixel_coords, color_image)

    # 绘制各个点和坐标信息
    for i in range(len(u_1)):
        cv2.circle(color_image, (u_1[i], v_1[i]), 5, (0, 0, 255), -1)  # 绘制蓝色小点
        x, y, z = selected_points[i]
        cv2.putText(color_image, f"({x:.2f}, {y:.2f}, {z:.2f})", (u_1[i] + 10, v_1[i] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return color_image

def process_bag_file(model, target_fps=30):
    """
    处理 .bag 文件并实时同步显示 YOLOv8 处理后的图像、原始深度图和生成的点云。

    Args:
        file_path (str): .bag 文件的路径。
        model (YOLO): 已加载的 YOLO 模型。
        target_fps (int, optional): 目标帧率。默认为 15。
    """
    pipeline, align = load_realsense_camera()  # 加载 bag 文件并初始化 pipeline 和 align 对象
    frame_time = 1.0 / target_fps  # 每帧的目标时间间隔
    if pipeline is None:
        print("Failed to start the pipeline.")
        return
    try:
        profile = pipeline.get_active_profile()
        streams = profile.get_streams()  # 获取流配置

        # 输出当前活动流的信息
        # 输出当前活动流的信息
        for stream in streams:
            print("Active Stream:", stream.stream_type(), "Resolution:", stream.as_video_stream_profile().width(), "x",
                  stream.as_video_stream_profile().height())

        # 创建 Open3D 可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Point Cloud')

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

            # 使用 YOLO 模型进行实例分割
            yolo_start = time.time()
            results = model(color_image, save=False)
            yolo_end = time.time()
            print(f"YOLO模型预测耗时 {yolo_end - yolo_start:.4f} seconds")

            # 检查模型是否检测到物体
            mask_process_start = time.time()
            if results and len(results) > 0:
                for result_idx, result in enumerate(results):
                    if result.masks is not None:
                        if hasattr(result.masks, 'data'):
                            masks = result.masks.data.cpu().numpy()  # 将掩码转换为 NumPy 数组
                        else:
                            raise TypeError(f"Unsupported mask object. Expected tensor-like, got {type(result.masks)}")

                        for mask_idx, mask in enumerate(masks):
                            # 根据掩码生成点云和对应的像素坐标
                            point_cloud, pixel_coords = get_point_cloud_from_mask(depth_frame, depth_intri, mask,
                                                                                  depth_image_shape)
                            mask_process_1=time.time()
                            print(f"掩码生成映射点云坐标耗时 "
                                  f"{mask_process_1 - mask_process_start:.4f} seconds")

                            # 使用 Open3D 显示点云
                            # if len(point_cloud) > 0:
                            #     cloud = o3d.geometry.PointCloud()
                            #     cloud.points = o3d.utility.Vector3dVector(point_cloud)
                            #     vis.clear_geometries()  # 清除之前的点云
                            #     vis.add_geometry(cloud)  # 添加新的点云
                            #     vis.poll_events()  # 更新窗口事件
                            #     vis.update_renderer()  # 更新渲染
                            mask_process_3=time.time()
                            print(f"显示点云标耗时 "
                                  f"{mask_process_3 - mask_process_1:.4f} seconds")

                            # 构造相机内参矩阵
                            fx = 386.888  # 焦距
                            fy = 386.477  # 焦距
                            cx = 321.010  # 光学中心X
                            cy = 246.484  # 光学中心Y
                            camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                            # 在 RGB 图像上绘制定位点
                            extract_and_draw_leftmost_points(point_cloud,color_image,camera_intrinsics)
                            mask_process_4=time.time()
                            print(f"RGB图像绘制点云点耗时 "
                                  f"{mask_process_4 - mask_process_3:.4f} seconds")

                    mask_process_end = time.time()
                    print(f"掩码匹配点云并生成二维点图像总耗时 "
                          f"{mask_process_end - mask_process_start:.4f} seconds")
                    # 显示 YOLO 处理后的分割图像
                    im_array = result.plot()
                    cv2.imshow('YOLO Instance Segmentation', im_array)
                    # 设置固定尺寸的可视化屏幕
                    # resized_frame = cv2.resize(color_image, (1920, 1080))
                    # cv2.imshow("Video", resized_frame)

                    cv2.imshow('Anchor point',color_image)

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

            # 限制帧率
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
    bag_file_path = "output.bag"
    process_bag_file(model)