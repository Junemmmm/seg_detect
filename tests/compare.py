import pyrealsense2 as rs
import numpy as np
from scipy.spatial import cKDTree
import torch
import cv2

def calculate_point_cloud_iou(points, ground_truth_points, threshold=0.01):
    """
    计算两个点云的交并比 (IoU)
    Args:
        points (np.ndarray): 预测点云，形状为 (N, 3)
        ground_truth_points (np.ndarray): 真值点云，形状为 (M, 3)
        threshold (float): 判断两点是否匹配的距离阈值，单位为米
    Returns:
        iou (float): 点云的交并比 (IoU)
    """
    tree_gt = cKDTree(ground_truth_points)
    tree_pred = cKDTree(points)

    distances_pred_to_gt, _ = tree_gt.query(points, k=1)
    match_pred_to_gt = distances_pred_to_gt < threshold

    distances_gt_to_pred, _ = tree_pred.query(ground_truth_points, k=1)
    match_gt_to_pred = distances_gt_to_pred < threshold

    intersection = np.sum(match_pred_to_gt) + np.sum(match_gt_to_pred)
    union = len(points) + len(ground_truth_points) - intersection
    iou = intersection / union if union > 0 else 0.0
    return iou

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

def process_bag_file(bag_file, threshold=0.01):
    """
    对 .bag 文件中的每一帧计算预测点云和真值点云的 IoU。
    Args:
        bag_file (str): .bag 文件路径
        threshold (float): 距离阈值，用于点云匹配
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file)
    pipeline.start(config)

    pc = rs.pointcloud()

    try:
        frame_count = 0
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            # 获取点云的顶点
            pc_data = pc.calculate(depth_frame)  # 获取点云数据
            vertices = np.asanyarray(pc_data.get_vertices())  # 将 BufData 转换为 NumPy 数组

            # 将点云数据从结构化数组中提取 x, y, z 坐标
            ground_truth_points = np.array([[v[0], v[1], v[2]] for v in vertices], dtype=np.float32)

            # 替换以下部分以生成预测点云
            mask = np.random.randint(0, 2, size=(depth_frame.height, depth_frame.width))  # 示例掩码
            points, _ = get_point_cloud_from_mask(depth_frame, depth_intrinsics, mask, (depth_frame.height, depth_frame.width))

            iou = calculate_point_cloud_iou(points, ground_truth_points, threshold)
            print(f"Frame {frame_count}: IoU = {iou:.4f}")
            frame_count += 1

    except RuntimeError:
        print("End of .bag file reached.")
    finally:
        pipeline.stop()



if __name__=='__main__':
    bag_file_path="output_chair2.bag"
    process_bag_file(bag_file_path)