import time
from ultralytics import YOLO
import cv2
import numpy as np


# 记录开始时间
start_time = time.time()

# 加载目标检测模型
det_model = YOLO('weights/yolov8n.pt')  # 目标检测模型路径

# 加载实例分割模型
seg_model = YOLO('weights/best.pt')  # 实例分割模型路径

# 记录模型加载时间
load_time = time.time()

# 读取图像
img = cv2.imread('data/11.png')

# 获取图像的尺寸
img_height, img_width = img.shape[:2]

# 对目标检测模型进行推理
det_results = det_model(img)
det_detections = det_results[0].boxes  # 获取检测结果的边界框
inference_time_1 = time.time()


# 对实例分割模型进行推理
seg_results = seg_model(img)

# 记录推理结束时间
inference_time_2 = time.time()

# 输出时间统计
print(f"Model loading time: {load_time - start_time:.2f} seconds")
print(f"Inference_1 time: {inference_time_1 - load_time:.2f} seconds")
print(f"Inference_2 time: {inference_time_2 - inference_time_1:.2f} seconds")


# 检查是否有掩膜
# if seg_results[0].masks is not None:
#     seg_masks = seg_results[0].masks.data.cpu().numpy()  # 实例分割掩膜
#
#     # 遍历掩膜并调整大小
#     for mask in seg_masks:
#         # 将掩膜调整为与输入图像相同的尺寸
#         mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
#         mask_resized = (mask_resized > 0.5).astype(np.uint8)  # 转换为二值掩膜
#
#         # 随机颜色
#         color = np.random.randint(0, 255, (3,), dtype=np.uint8)
#
#         # 绘制半透明掩膜
#         img[mask_resized == 1] = img[mask_resized == 1] * 0.5 + color * 0.5

# 绘制目标检测结果
for detection in det_detections:
    x1, y1, x2, y2 = detection.xyxy[0]  # 获取边界框坐标
    conf = detection.conf[0]  # 获取置信度
    cls = int(detection.cls[0])  # 获取类别

    # 绘制边界框和标签
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    cv2.putText(img, f"{det_results[0].names[cls]} {conf:.2f}", (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


# 显示结果
cv2.imshow("YOLOv8 Detection and Segmentation", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

