import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

def draw_sparse_pointcloud_mask_overlay(color_image, depth_frame, mask, depth_intrinsics, sample_ratio=0.15):
    """
    在color_image上，将掩码mask区域随机稀疏采样一定比例的点，
    用鲜艳颜色（红、蓝、黄）的小圆点覆盖，点大小和深度相关。

    sample_ratio: 采样点占掩码点总数的比例，控制稀疏度，0~1之间
    """
    overlay_img = color_image.copy()
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return overlay_img

    # 颜色列表，BGR格式
    colors = [(0,0,255),   # 红色
              (255,0,0),   # 蓝色
              (0,255,255)] # 黄色

    sample_count = max(1, int(len(xs) * sample_ratio))
    indices = np.random.choice(len(xs), size=sample_count, replace=False)
    xs_sampled = xs[indices]
    ys_sampled = ys[indices]

    for i, (x, y) in enumerate(zip(xs_sampled, ys_sampled)):
        depth = depth_frame.get_distance(x, y)
        if 0 < depth < 20:  # 过滤无效/远点
            radius = int(np.clip(20 / (depth * 10), 1, 4))
            base_color = colors[i % len(colors)]  # 循环选择颜色
            # 按深度调整颜色亮度
            intensity_scale = np.clip((5 - depth) / 5, 0.3, 1.0)  # 0.3~1.0范围内亮度调节
            color = tuple(int(c * intensity_scale) for c in base_color)
            cv2.circle(overlay_img, (x, y), radius, color, -1)

    return overlay_img


def draw_mask_overlay(color_image, mask, alpha=0.4, color=(0, 255, 0)):
    """将掩码以半透明形式覆盖在彩色图上"""
    overlay = color_image.copy()
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return overlay

def process_bag_file(file_path, seg_model, output_video_path=None):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(file_path, repeat_playback=False)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # 视频写入相关
    video_writer = None
    frame_size = (640, 480)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()

            results = seg_model(color_image)
            result = results[0]

            mask = None
            if result.masks is not None and hasattr(result.masks, "data"):
                masks = result.masks.data.cpu().numpy()
                if len(masks) > 0:
                    max_mask_idx = np.argmax([m.sum() for m in masks])
                    mask = (masks[max_mask_idx] > 0.5).astype(np.uint8)

            if mask is not None:
                pointcloud_img = draw_sparse_pointcloud_mask_overlay(color_image, depth_frame, mask, depth_intrinsics, sample_ratio=0.15)
                mask_overlay_img = draw_mask_overlay(color_image, mask, alpha=0.4, color=(0, 255, 0))
            else:
                pointcloud_img = color_image.copy()
                mask_overlay_img = color_image.copy()

            cv2.imshow("PointCloud Mask Overlay", pointcloud_img)
            cv2.imshow("Instance Segmentation Mask Overlay", mask_overlay_img)

            if output_video_path:
                if video_writer is None:
                    # 两个窗口放一起录制（横向拼接）
                    w, h = frame_size
                    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w*2, h))
                combined_frame = np.hstack((pointcloud_img, mask_overlay_img))
                video_writer.write(combined_frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC退出
                break

    finally:
        pipeline.stop()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    seg_model = YOLO('weights/best1.pt')  # 本地模型路径
    bag_file_path = "new1.bag"   # 本地bag文件路径
    process_bag_file(bag_file_path, seg_model, output_video_path='output_result.mp4')
