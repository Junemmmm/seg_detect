import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import open3d as o3d  # 用于显示点云
import time
import pandas as pd
from scipy.interpolate import make_interp_spline
from method.method_pre import project_to_image
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

"""
根据控制点绘制贝塞尔曲线
Args: 
    控制点像素下坐标、RGB图像
Returns: 
    绘制好贝塞尔曲线的RGB图像
"""
def draw_bezier_curve(pixel_coords, color_image, threshold=3):
    """
    使用贝塞尔曲线插值，并绘制插值曲线，同时将内点标为白色，外点标为红色。

    参数：
    - pixel_coords: 点云坐标，形状为 (N, 2) 的数组。
    - color_image: 输入彩色图像，用于绘制曲线。
    - threshold: 残差阈值，区分内点和外点。

    返回：
    - 绘制了插值曲线和内/外点的图像。
    """
    # 当点云点数不足时返回原图像
    if len(pixel_coords) < 2:
        print("Not enough points to draw a curve.")
        return color_image  # 返回原图像

    # 提取 X 和 Y 坐标
    x_points = pixel_coords[:, 0]
    y_points = pixel_coords[:, 1]

    # 判断点的数量
    num_points = len(x_points)
    if num_points < 2:
        print("Not enough points to draw a curve.")
        return color_image

    # 动态选择插值阶数，确保不会超过点数 - 1
    k = min(3, num_points - 1)

    try:
        # 贝塞尔曲线插值
        t = np.linspace(0, 1, num_points)  # 参数化
        spl_x = make_interp_spline(t, x_points, k=k)  # 插值器
        spl_y = make_interp_spline(t, y_points, k=k)  # 插值器

        # 生成插值点
        t_new = np.linspace(0, 1, 100)
        x_smooth = spl_x(t_new)
        y_smooth = spl_y(t_new)

        # 计算原始点到插值曲线的残差
        fitted_x = spl_x(t)  # 原始 t 对应的插值 x 值
        fitted_y = spl_y(t)  # 原始 t 对应的插值 y 值
        residuals = np.sqrt((x_points - fitted_x) ** 2 + (y_points - fitted_y) ** 2)  # 欧氏距离

        # # 根据残差区分内点和外点
        # inlier_mask = residuals <= threshold  # 内点：残差小于阈值
        # outlier_mask = ~inlier_mask  # 外点：残差大于阈值
        #
        # # 绘制内点和外点
        # for idx, (x, y) in enumerate(zip(x_points, y_points)):
        #     if inlier_mask[idx]:
        #         cv2.circle(color_image, (int(x), int(y)), 5, (255, 255, 255), -1)  # 白色内点
        #     else:
        #         cv2.circle(color_image, (int(x), int(y)), 5, (0, 0, 0), -1)  # 红色外点

        # 绘制贝塞尔曲线
        for i in range(len(x_smooth) - 1):
            cv2.line(color_image, (int(x_smooth[i]), int(y_smooth[i])),
                     (int(x_smooth[i + 1]), int(y_smooth[i + 1])), (0, 255, 0), 2)  # 绘制绿色轨迹线

    except ValueError as e:
        print(f"Error in spline interpolation: {e}")
        return color_image  # 如果插值失败，返回原图像

    return color_image

"""
根据控制点使用最小二乘法绘制曲线
"""
import numpy as np
import cv2

def draw_least_squares_curve(pixel_coords, color_image, threshold=5):
    """
    使用最小二乘法拟合曲线，并绘制拟合结果，同时将内点标为白色，外点标为红色。

    参数：
    - pixel_coords: 点云坐标，形状为 (N, 2) 的数组。
    - color_image: 输入彩色图像，用于绘制曲线。
    - threshold: 残差阈值，区分内点和外点。

    返回：
    - 绘制了拟合曲线和内/外点的图像。
    """
    # 当点云点数不足时返回原图像
    if len(pixel_coords) < 2:
        print("Not enough points to draw a curve.")
        return color_image  # 返回原图像

    # 提取 X 和 Y 坐标
    x_points = pixel_coords[:, 0]
    y_points = pixel_coords[:, 1]

    # 判断点的数量
    num_points = len(x_points)
    if num_points < 2:
        print("Not enough points to draw a curve.")
        return color_image

    # 动态选择拟合多项式的阶数，最高为三阶
    degree = min(3, num_points - 1)

    try:
        # 最小二乘法多项式拟合
        coefficients = np.polyfit(x_points, y_points, degree)  # 拟合曲线的多项式系数
        polynomial = np.poly1d(coefficients)  # 生成多项式函数

        # 计算残差：实际值与拟合值的差
        y_fitted = polynomial(x_points)  # 对原始点的 x 值进行拟合
        residuals = np.abs(y_points - y_fitted)  # 计算残差

        # 根据残差区分内点和外点
        inlier_mask = residuals <= threshold  # 内点：残差小于阈值
        outlier_mask = ~inlier_mask  # 外点：残差大于阈值

        # 绘制内点和外点
        for idx, (x, y) in enumerate(zip(x_points, y_points)):
            if inlier_mask[idx]:
                cv2.circle(color_image, (int(x), int(y)), 4, (255, 255, 255), -1)  # 白色内点
            else:
                cv2.circle(color_image, (int(x), int(y)), 4, (0, 0, 255), -1)  # 红色外点

        # 生成拟合曲线的点
        x_smooth = np.linspace(x_points.min(), x_points.max(), 100)  # 细化 x 点
        y_smooth = polynomial(x_smooth)  # 计算对应的 y 点

        # 绘制拟合曲线
        for i in range(len(x_smooth) - 1):
            cv2.line(color_image, (int(x_smooth[i]), int(y_smooth[i])),
                     (int(x_smooth[i + 1]), int(y_smooth[i + 1])), (0, 255, 0), 2)  # 绘制绿色曲线

    except np.linalg.LinAlgError as e:
        print(f"Error in least squares fitting: {e}")
        return color_image  # 如果拟合失败，返回原图像

    return color_image

"""
根据控制点使用RANSAC法绘制曲线
"""
def draw_ransac_curve(pixel_coords, color_image):
    # 当点云点数不足时返回原图像
    if len(pixel_coords) < 2:
        print("Not enough points to draw a curve.")
        return color_image  # 返回原图像

    # 提取 X 和 Y 坐标
    x_points = pixel_coords[:, 0].reshape(-1, 1)  # 转为二维数组，适应 RANSAC 接口
    y_points = pixel_coords[:, 1]

    # 判断点的数量
    num_points = len(x_points)
    if num_points < 2:
        print("Not enough points to draw a curve.")
        return color_image

    try:
        # 创建 RANSAC 模型，内嵌多项式回归（最高三阶）
        degree = 3  # 多项式的最高阶
        model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor())

        # 拟合 RANSAC 模型
        model.fit(x_points, y_points)
        ransac_inliers = model.named_steps['ransacregressor'].inlier_mask_

        # 生成拟合曲线点
        x_smooth = np.linspace(x_points.min(), x_points.max(), 100).reshape(-1, 1)  # 平滑的 x 坐标
        y_smooth = model.predict(x_smooth)  # 对应的 y 值

        # 绘制 RANSAC 拟合曲线
        for i in range(len(x_smooth) - 1):
            cv2.line(color_image, (int(x_smooth[i]), int(y_smooth[i])),
                     (int(x_smooth[i + 1]), int(y_smooth[i + 1])), (0, 255, 0), 2)  # 绿色拟合曲线

        # 可选：绘制内点（白色）和外点（红色）
        for idx, inlier in enumerate(ransac_inliers):
            point_color = (255, 255, 255) if inlier else (0, 0, 255)  # 白色内点，红色外点
            cv2.circle(color_image, (int(x_points[idx][0]), int(y_points[idx])), 4, point_color, -1)

    except Exception as e:
        print(f"Error in RANSAC fitting: {e}")
        return color_image  # 如果拟合失败，返回原图像

    return color_image


"""
    从点云中按照区间分段提取X轴绝对值最小的点并返回它们的坐标(即)。
    Args:
        输入的点云数据(形状为 (N, 3)), RGB图像， 相机的内参矩阵
    Returns:
        图像上绘制每个区段中X轴绝对值最小的点及坐标信息，以及根据边沿点生成的贝塞尔曲线
"""
def extract_and_draw_leftmost_points(point_cloud, color_image, camera_intrinsics):
    # 检查 color_image 是否为灰度图像（2D）
    if len(color_image.shape) == 2:
        print("Warning: color_image is a grayscale image, converting to RGB.")
        # 如果是灰度图像，将其转换为 3 通道的 RGB 图像
        color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2BGR)
    # 检查 point_cloud 是否为空
    if len(point_cloud) == 0:
        print("Warning: point_cloud is empty.")
        return color_image  # 如果点云为空，直接返回原图像

    # 转换为 DataFrame 处理
    cloud_df = pd.DataFrame(point_cloud, columns=['x', 'y', 'z'])
    # 检查云数据是否有效
    if cloud_df.empty:
        print("Warning: Cloud data is empty after conversion.")
        return color_image
    """
    绘制深度值最小的点
    """
    # 对深度值进行升序排序，选择倒数第三到倒数第十的点
    sorted_depth_points = cloud_df.nsmallest(700, 'z')  # 获取深度值最小的前500个点
    # 获取深度值最大的前50个点
    sorted_depth_points = sorted_depth_points.nlargest(50, 'z')  # 获取深度最大的前50个点


    # 在这些点中找到 X 轴绝对值最小的点
    min_x_abs_point = sorted_depth_points .loc[sorted_depth_points ['x'].abs().idxmin()]

    # 提取该点的 3D 坐标
    min_depth_coord = [min_x_abs_point['x'], min_x_abs_point['y'], min_x_abs_point['z']]

    # 投影深度最小点到图像平面
    min_u, min_v = project_to_image(np.array([min_depth_coord]), camera_intrinsics)
    # 打印投影坐标，验证是否正确
    print(f"Projected Coordinates: min_u = {min_u}, min_v = {min_v}")

    # 获取图像的宽度和高度
    height, width, _ = color_image.shape

    # 最近点Nearest Point文本内容
    text = f"Nearest Point: ({min_depth_coord[0]:.2f}, {min_depth_coord[1]:.2f}, {min_depth_coord[2]:.2f})"
    # 计算文本的宽度和高度
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # 计算起始位置 (u, v)，并考虑偏移
    x = min_u[0] + 10
    y = min_v[0] + 10

    # 确保文本不会超出图像范围
    x = max(0, min(x, width - text_width - 5))  # 留出 5 像素的边距
    y = max(text_height + 10, min(y, height - 5))  # 留出 5 像素的边距，确保文本不在顶部被截断

    # 绘制文本
    cv2.putText(color_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    '''
        按照深度值z进行分段，生成6个区间并选取X轴绝对值最小的点作为控制点
    '''
    # 生成4个深度区间并去除重复边界
    depth_bins = np.linspace(cloud_df['z'].min(), cloud_df['z'].max(), 5)
    depth_bins = np.unique(depth_bins)  # 去除重复的边界

    # 如果边界值足够唯一，使用 pd.cut 进行分段
    if len(depth_bins) > 1:
        cloud_df['depth_bin'] = pd.cut(cloud_df['z'], bins=depth_bins, duplicates='drop')
    else:
        print("Warning: depth_bins has too few unique values.")
        return color_image

    selected_points = []
    bin_width = depth_bins[1] - depth_bins[0]  # 计算每个深度区间的宽度

    for bin in cloud_df['depth_bin'].unique():
        bin_points = cloud_df[cloud_df['depth_bin'] == bin]  # 提取当前深度段的点

        if not bin_points.empty:
            # 计算当前深度区间的中间一小段区间
            bin_min = bin.left  # 当前区间的最小值
            bin_max = bin.right  # 当前区间的最大值
            middle_range_min = bin_min + bin_width * 0.25  # 中间区间的最小值
            middle_range_max = bin_max - bin_width * 0.25  # 中间区间的最大值

            # 在中间区间范围内选择点
            middle_bin_points = bin_points[
                (bin_points['z'] >= middle_range_min) & (bin_points['z'] <= middle_range_max)]

            if not middle_bin_points.empty:
                # 获取 X 轴绝对值最小的点
                rightmost_point = middle_bin_points.loc[middle_bin_points['x'].abs().idxmin()]
                selected_points.append([rightmost_point['x'], rightmost_point['y'], rightmost_point['z']])

    # 如果没有选中任何点，返回原图像
    if len(selected_points) == 0:
        print("Warning: No points were selected.")
        return color_image  # 没有选中任何点时返回原图像

    # 将选择的点转换为 NumPy 数组
    selected_points = np.array(selected_points)

    # 投影到图像平面
    u_1, v_1 = project_to_image(selected_points, camera_intrinsics)

    '''
        按照深度值z进行分段，生成16个区间并选取X轴绝对值最小的点作为贝塞尔曲线绘制点
    '''
    # 按照深度值z进行分段，并选取X轴绝对值最小的点作为控制点
    depth_curb = np.linspace(cloud_df['z'].min(), cloud_df['z'].max(), 16)  # 生成16个区间
    cloud_df['depth_curb'] = pd.cut(cloud_df['z'], bins=depth_curb)

    # 确保 depth_curb 至少有两个不同的边界
    if len(depth_curb) > 1:
        cloud_df['depth_curb'] = pd.cut(cloud_df['z'], bins=depth_curb, duplicates='drop')
    else:
        print("Warning: depth_curb has too few unique values.")
        return color_image

    depth_curbPoints = []
    for bin in cloud_df['depth_curb'].unique():
        bin_points = cloud_df[cloud_df['depth_curb'] == bin]  # 提取当前深度段的点
        if not bin_points.empty:
            # 获取X绝对值最小的点
            rightmost_point = bin_points.loc[bin_points['x'].abs().idxmin()]
            depth_curbPoints.append([rightmost_point['x'], rightmost_point['y'], rightmost_point['z']])

    # **将深度最小点加入贝塞尔曲线控制点**
    depth_curbPoints.append(min_depth_coord)

    # 转换为数组
    depth_curbPoints = np.array(depth_curbPoints)

    # 使用相机内参矩阵将3D点投影到图像上
    u, v = project_to_image(depth_curbPoints, camera_intrinsics)

    # 使用投影后的像素坐标绘制贝塞尔曲线
    pixel_coords = np.column_stack((u, v))  # 将u, v坐标合并为 (u, v) 对
    stime = time.time()
    """
    绘制贝塞尔曲线
    """
    color_image = draw_bezier_curve(pixel_coords, color_image)
    edtime=time.time()
    draw_time=edtime-stime
    print(f"绘制拟合曲线总耗时{draw_time:.4f}seconds")

    # 绘制各个点和坐标信息
    for i in range(len(u_1)):
        cv2.circle(color_image, (u_1[i], v_1[i]), 5, (0, 0, 255), -1)  # 绘制红色小点
        x, y, z = selected_points[i]
        cv2.putText(color_image, f"({x:.2f}, {y:.2f}, {z:.2f})", (u_1[i] + 10, v_1[i] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # **在控制点中单独标记深度最小点**
    cv2.circle(color_image, (min_u[0], min_v[0]), 5, (0, 0, 255), -1)  # 紫色，区分于贝塞尔曲线控制点

    return color_image


"""
统计滤波基于局部密度，通过计算每个点的邻居数，剔除噪声点。若一个点的邻居数低于预设阈值，则该点可能是噪声。
此方法适合去除孤立或稀疏点。可以设置邻居数量和距离阈值，以去除较远的孤立点。
"""
def apply_statistical_outlier_removal(point_cloud):
    """
    输入: np.ndarray (N,3) 或 open3d.geometry.PointCloud
    输出: np.ndarray (M,3)，滤波后的点云
    """
    # 如果是 numpy，先转成 PointCloud
    if isinstance(point_cloud, np.ndarray):
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud)
    else:
        o3d_cloud = point_cloud

    # 执行统计滤波
    cl, ind = o3d_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    filtered_pcd = o3d_cloud.select_by_index(ind)

    # 转 numpy
    return np.asarray(filtered_pcd.points)

"""
使用聚类滤波算法（如DBSCAN）划分点云，选择点数最多的聚类作为主要部分，过滤掉其他小聚类。
DBSCAN可直接应用在Open3D点云中，剔除分离较远的聚类。
"""


def apply_dbscan_filter(point_cloud, eps=0.05, min_points=10):
    """
    输入: np.ndarray (N,3) 或 open3d.geometry.PointCloud
    输出: np.ndarray (M,3)，保留最大簇后的点云
    """
    # 判断输入类型
    if isinstance(point_cloud, o3d.geometry.PointCloud):
        points = np.asarray(point_cloud.points)
    else:
        points = np.asarray(point_cloud)

    if points.shape[0] == 0:
        print("Warning: 输入点云为空")
        return np.empty((0, 3))

    # 构建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # DBSCAN 聚类
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    if labels.max() < 0:
        print("DBSCAN未找到任何簇, 返回原始点云")
        return points  # 直接返回原始 numpy

    # 找到最大簇
    largest_cluster = np.argmax(np.bincount(labels[labels >= 0]))
    indices = np.where(labels == largest_cluster)[0]

    # 返回 numpy
    filtered_pcd = pcd.select_by_index(indices)
    return np.asarray(filtered_pcd.points)
