import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from ultralytics import YOLO

# 加载已经训练好的YOLOv8-seg模型
model = YOLO('weights/best.pt')  # 使用你的训练好的模型路径

# 读取图片
img_path = 'data/shelter2.png'  # 你的测试图片路径
img = Image.open(img_path)

# 推理
results = model(img)

# 获取推理结果
result = results[0]  # 结果列表中的第一个结果

# 提取掩码、边界框、标签和置信度
masks = result.masks.xy  # 掩码坐标是列表类型，无需调用 .cpu()
boxes = result.boxes.xywh.cpu()  # 确保边界框在 CPU 上
labels = result.names  # 标签名称
class_ids = result.boxes.cls.cpu().numpy().astype(int)  # 类别ID，确保在 CPU 上
confidences = result.boxes.conf.cpu().numpy()  # 置信度，确保在 CPU 上


# 自定义颜色 (颜色值应该在 0-1 范围内)
def random_color():
    return [random.randint(0, 255) / 255 for _ in range(3)]


# 创建一个空的画布
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(np.array(img))

# 绘制每个目标的掩码、边界框和标签
for i, mask in enumerate(masks):
    color = random_color()
    # 绘制掩码
    mask_polygon = np.array(mask).reshape(-1, 2)
    ax.fill(mask_polygon[:, 0], mask_polygon[:, 1], color=color, alpha=0.5)

    # 绘制边界框
    box = boxes[i]
    x1, y1, w, h = box
    rect = plt.Rectangle((x1 - w / 2, y1 - h / 2), w, h, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

    # 获取类别名称和置信度
    label = labels[class_ids[i]]
    confidence = confidences[i]
    confidence = confidences[i] * 100  # 将置信度转换为百分比

    # 将标签信息绘制在边界框的左上角
    ax.text(x1 - w / 2, y1 - 5, f'{label} {confidence:.2f}%', color=color, fontsize=12, ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.7))

plt.axis('off')
plt.show()
