import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import open3d as o3d  # 用于显示点云
import time
import pandas as pd
from scipy.interpolate import make_interp_spline


"""
提取相机内参
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
创建一个 RealSense 数据流（pipeline）和配置（config）。
输入 .bag 文件进行加载。
输出 pipeline用于处理数据流
输出 glign用于在后续处理中对齐深度和颜色图像数据
"""
def load_bag_file(file_path):
    # 实例化 RealSense pipeline，负责管理从摄像头或 .bag 文件中捕获和处理数据流
    pipeline = rs.pipeline()
    config = rs.config()  # 用于配置pipeline 中的数据流方式，可以用来选择数据流类型（例如深度、颜色、红外等）
    # 从 .bag 文件中加载数据，不重复播放
    config.enable_device_from_file(file_path, repeat_playback=False)
    pipeline.start(config)  # 启动 pipeline，开始从指定配置中获取数据

    align_to = rs.stream.color  # 设置对齐的目标流类型为颜色图像
    align = rs.align(align_to)  # 创建一个对齐对象，用于对齐深度图像到颜色图像
    # pipeline用于处理数据流, align用于在后续处理过程中对齐深度和颜色图像数据，确保像素点在不同图像流中的一致性。
    return pipeline, align

"""
将点云数据保存为 PLY 文件。
Args:
    filename (str): 输出 PLY 文件的路径。
    point_cloud (np.ndarray): 点云数据，形状为 (N, 3)。
"""
def save_point_cloud_to_ply(filename, point_cloud):
    # 过滤掉不需要的点
    filtered_points = [point for point in point_cloud if
                       not (point[0] == 0.0 and point[1] == 0.0 and point[2] == 0.0) and not (
                                   point[0] == -0.0 and point[1] == -0.0 and point[2] == 0.0)]

    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(filtered_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in filtered_points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

"""
从RGB掩码中生成点云
Args: 
    深度帧、深度相机内、掩码、深度图像shape
Returns: 
    Numopy数组格式的 点云坐标、像素坐标
"""
import numpy as np
import torch
import cv2
import pyrealsense2 as rs

"""
体素化法三维重构
"""
def get_voxelized_point_cloud_from_mask(depth_frame, depth_intri, mask, depth_image_shape, voxel_size=0.05):
    # 检查掩码是否为 PyTorch Tensor，如果是则转换为 NumPy 数组
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()  # 将 Tensor 转换为 NumPy 数组格式
    elif isinstance(mask, np.ndarray):
        mask_np = mask  # 如果掩码已经是 NumPy 数组，直接使用
    else:
        raise TypeError("Unsupported mask type. Expected torch.Tensor or numpy.ndarray.")

    # 判断 mask_np 是否是二维数组
    if len(mask_np.shape) != 2:
        raise ValueError(f"Expected 2D mask, but got shape {mask_np.shape}")

    # 调整掩码到深度图像尺寸
    if mask_np.shape != depth_image_shape:
        # OpenCV 的尺寸顺序为 (width, height)，因此要进行逆序调整
        mask_np = cv2.resize(mask_np, depth_image_shape[::-1], interpolation=cv2.INTER_LINEAR)

    # 找到掩码中为 True 或非零的像素点坐标索引
    mask_indices = np.argwhere(mask_np)

    # 初始化存储体素坐标的集合
    voxel_set = set()

    for idx in mask_indices:
        uy, ux = idx  # 图像中的 y, x 坐标
        dis = depth_frame.get_distance(ux, uy)  # 获取点的深度值

        # 当拥有深度信息时才会进行记录
        if dis > 0:
            # 将像素坐标和深度值转化为相机坐标系下的 (x, y, z) 三维点
            point = rs.rs2_deproject_pixel_to_point(depth_intri, [ux, uy], dis)

            # 计算该点所在的体素坐标
            voxel_x = int(np.floor(point[0] / voxel_size))
            voxel_y = int(np.floor(point[1] / voxel_size))
            voxel_z = int(np.floor(point[2] / voxel_size))

            # 将体素坐标添加到集合中
            voxel_set.add((voxel_x, voxel_y, voxel_z))

    # 转换体素集合为数组格式
    voxel_grid = np.array(list(voxel_set))
    return voxel_grid  # 返回体素化后的三维点云


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
        # OPENCV是 w x h ，而depth_image_shape 是 h x w（切片操作 逆序）统一到 w x h 再使用插值算法双线性插值方法进行resize
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

def handle_frame(pipeline,align):

    frames = pipeline.wait_for_frames()  # 等待并获取当前帧
    aligned_frames = align.process(frames)  # 对齐深度和颜色帧
    depth_frame = aligned_frames.get_depth_frame()  # 获取对齐后的深度帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐后的颜色帧

    # 如果没有有效的深度帧或颜色帧，则跳过
    if not depth_frame or not color_frame:
        print("Skipping frame due to missing depth or color data.")

    # 3. 转换深度图像和颜色图像为 NumPy 数组
    conversion_start = time.time()
    depth_image = np.asanyarray(depth_frame.get_data())  # 将深度数据转换为 NumPy 数组
    color_image = np.asanyarray(color_frame.get_data())  # 将颜色数据转换为 NumPy 数组
    depth_intri = depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度相机的内参
    depth_image_shape = (depth_frame.get_height(), depth_frame.get_width())
    conversion_end = time.time()
    print(f"帧转数组耗时 {conversion_end - conversion_start:.4f} seconds")