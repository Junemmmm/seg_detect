"""
次代码勇于实现视频分帧实例分割物体
将RGB图像与点云图像匹配
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO  # 将YOLOv8导入到该py文件中


# 加载官方或自定义模型
model = YOLO("weights/best.pt")  # 加载一个官方的检测模型
# model = YOLO(r"E:\Deep learning\YOLOv8\yolov8s.pt")  # 加载一个官方的检测模型
# model = YOLO(r"E:\Deep learning\YOLOv8\yolov8n-seg.pt")  # 加载一个官方的分割模型
# model = YOLO(r"E:\Deep learning\YOLOv8\yolov8n-pose.pt")  # 加载一个官方的姿态模型


# 深度相机配置
pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 初始化摄像头深度流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe_profile = pipeline.start(config)  # 启用管段流
align = rs.align(rs.stream.color)  # 这个函数用于将深度图像与彩色图像对齐

"""
get_aligned_images()函数：
返回 深度帧的相机内参 深度帧 彩色帧转换的numpy数组
"""
def get_aligned_images():  # 定义一个获取图像帧的函数，返回深度和彩色数组
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐
    depth_frame = aligned_frames.get_depth_frame()  # 获取深度帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧
    depth_image = np.asanyarray(depth_frame.get_data())  # 将深度帧转换为NumPy数组
    color_image = np.asanyarray(color_frame.get_data())  # 将彩色帧转化为numpy数组

    # 获取深度帧的相机内参。包含相机的焦距、主点等信息，主要用于将像素坐标转换为实际的物理坐标
    depth_intri = depth_frame.profile.as_video_stream_profile().intrinsics
    # 获取彩色内参
    color_intri = color_frame.profile.as_video_stream_profile().intrinsics
    # 命令行输出内参检查
    # print("Depth Intrinsics:",depth_intri)
    # print("Color Intrinsics:",color_intri)

    # cv2.applyColorMap（）将深度图像转化为彩色图像，以便更好的可视化分析
    # 将深度图像转换为伪彩色图像
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.07), cv2.COLORMAP_JET)
    # 返回深度内参、对齐深度帧、彩色图像
    return depth_intri, depth_frame, color_image

"""
get_point_cloud_from_mask函数：
遍历RGB的掩膜中的每一个像素点，获取掩膜的点云
"""
def get_point_cloud_from_mask(depth_frame,depth_intri,mask):
    points = []

    # 遍历掩码图像中的每一个像素
    mask_indices = np.argwhere(mask)  # 获取掩码为True的所有像素坐标
    for idx in mask_indices:
        uy, ux = idx  # 图像中的 y, x 坐标（注意顺序）

        # 获取该像素点的深度值
        dis = depth_frame.get_distance(ux, uy)
        if dis > 0:  # 确保深度值有效
            # 将像素坐标和深度值转化为相机坐标系下的 (x, y, z) 三维点
            point = rs.rs2_deproject_pixel_to_point(depth_intri, [ux, uy], dis)
            points.append(point)
    # 返回点云作为 NumPy 数组
    return np.array(points)

"""

"""
def process_segmentation_and_point_cloud():
    depth_intri, depth_frame, color_image = get_aligned_images()  # 获取对齐后的深度和彩色帧
    source = [color_image]

    # 调用实例分割模型
    results = model.segment(source, save=False)

    # 遍历每个检测到的目标
    for result in results:
        boxes = result.boxes.xywh.tolist()  # 获取目标边框列表
        masks = result.masks.cpu().numpy()  # 获取分割掩码

        # 遍历每个目标的掩码
        for mask in masks:
            # 根据当前目标的分割掩码和深度帧生成点云
            point_cloud = get_point_cloud_from_mask(depth_frame, depth_intri, mask)

            # 打印点云信息，可以做进一步处理，比如保存为文件或可视化
            print(f"生成的点云：{point_cloud.shape[0]} 个点")
            print(f"部分点云数据: {point_cloud[:5]}")  # 打印前五个点

            # 可选：将点云保存到文件（如 PLY 格式）
            save_point_cloud_to_ply("output.ply", point_cloud)

def save_point_cloud_to_ply(filename, point_cloud):
    """
    将点云数据保存为 PLY 文件
    :param filename: 文件名
    :param point_cloud: 点云数据，NumPy数组，每行是 (x, y, z) 坐标
    """
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


if __name__ == '__main__':
    process_segmentation_and_point_cloud()