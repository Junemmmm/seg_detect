import cv2
import numpy as np
from ultralytics import YOLO
from scipy.interpolate import splprep, splev

# === 加载模型 ===
model = YOLO("weights/best1.pt")  # 替换为你的模型路径
image = cv2.imread("data/rgb_2024110335_1089.png")  # 替换为你的图像路径
original_h, original_w = image.shape[:2]

# === 模型推理 ===
results = model(image)[0]

# === 图像副本 ===
image_seg = image.copy()       # 实例分割结果图像
image_edge = image.copy()      # 拟合曲线图像

# === 可视化全部分割掩码到 image_seg ===
if results.masks is not None:
    masks = results.masks.data.cpu().numpy()  # shape: [N, H, W]
    print(f"Total masks: {len(masks)}")

    for mask in masks:
        color_mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_seg, contours, -1, (255, 0, 0), 2)  # 蓝色画出所有实例

    # === 仅取最大实例用于边缘拟合 ===
    mask_areas = [np.sum(mask > 0.5) for mask in masks]
    max_idx = np.argmax(mask_areas)
    largest_mask = masks[max_idx]
    mask_uint8 = (largest_mask * 255).astype(np.uint8)

    # === 找到掩码内的像素点 ===
    ys, xs = np.where(mask_uint8 > 127)
    if len(ys) == 0:
        print("Selected instance has no pixels.")
        exit()

    # 转为笛卡尔坐标系（左下角为原点）
    ys_flipped = original_h - ys
    points = np.vstack((xs, ys_flipped)).T

    # 分段处理（Y轴方向16段）
    y_min, y_max = ys_flipped.min(), ys_flipped.max()
    y_bins = np.linspace(y_min, y_max, 30)

    edge_points = []
    for i in range(29):
        y_start, y_end = y_bins[i], y_bins[i+1]
        segment_mask = (ys_flipped >= y_start) & (ys_flipped < y_end)
        segment_points = points[segment_mask]

        if len(segment_points) == 0:
            continue

        # 选取X值最小的点（相对于左下角原点）
        # min_idx = np.argmin(segment_points[:, 0])
        max_idx = np.argmax(segment_points[:, 0])
        edge_points.append(segment_points[max_idx])

    # 拟合贝塞尔曲线
    edge_points = np.array(edge_points)
    if len(edge_points) >= 4:
        edge_points = edge_points[np.argsort(edge_points[:, 1])]  # 按Y升序排序
        tck, _ = splprep(edge_points.T, s=3)
        u_fine = np.linspace(0, 1, 200)
        bezier = splev(u_fine, tck)
        bezier_curve = np.vstack(bezier).T

        # 转回图像坐标系（OpenCV Y 向下）
        bezier_curve[:, 1] = original_h - bezier_curve[:, 1]

        # 绘制贝塞尔曲线到 image_edge
        for i in range(1, len(bezier_curve)):
            pt1 = tuple(np.round(bezier_curve[i-1]).astype(int))
            pt2 = tuple(np.round(bezier_curve[i]).astype(int))
            cv2.line(image_edge, pt1, pt2, (0, 255, 0), 2)

        # 可选：画出分段选中的边缘点（调试用）
        # for pt in edge_points:
        #     x, y = int(pt[0]), int(original_h - pt[1])
        #     cv2.circle(image_edge, (x, y), 4, (0, 255, 255), -1)  # 黄色圆点

    else:
        print("Not enough points to fit Bézier curve.")

else:
    print("No mask detected.")

# === 显示结果 ===
cv2.imshow("Instance Segmentation", image_seg)
cv2.imshow("Fitted Edge Line", image_edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

# === 你也可以保存图像 ===
cv2.imwrite("segmentation_result.jpg", image_seg)
cv2.imwrite("edge_fitting_result.jpg", image_edge)

