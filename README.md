# seg_detect —— 多任务目标检测与路沿边缘检测系统

本项目面向智能轮椅与辅助驾驶等场景，基于 Intel RealSense D455 深度相机设计了一套 RGB-D 路沿检测系统，融合多任务深度学习算法与三维点云算法：

- **目标检测定位**：利用 YOLOv8 目标检测模型识别行人、车辆等目标，并结合深度信息完成其三维定位；
- **路沿边缘检测定位**：利用 YOLOv8 实例分割模型分割路沿，将分割掩码映射为点云，经统计滤波与 DBSCAN 聚类滤波去除噪声后，按深度分区间提取边缘控制点并拟合贝塞尔曲线，实现路沿边缘的准确定位。

两路任务并行运行、结果实时叠加显示。经过光照变化、遮挡、多路况等复杂环境的多次测试，系统检测准确、稳定，平均处理速度约 35 FPS。此外，为满足嵌入式部署需求，本项目还规划了基于 ZYNQ7015 异构 SoC 的 INT8 量化与硬件加速方案。

## 整体流程

1. 读取 RealSense 录制的 `.bag` 文件（或实时相机流）
2. 用 YOLOv8 实例分割模型识别路沿，得到掩码
3. 将分割掩码映射到三维点云（`rs2_deproject_pixel_to_point`）
4. 点云滤波（统计滤波 + DBSCAN 聚类滤波）
5. 按深度分区间提取边缘控制点
6. 用贝塞尔曲线 / 最小二乘 / RANSAC 拟合路沿轮廓
7. 投影回 RGB 图像可视化，并可用目标检测模型做中心点定位

## 目录结构

```
seg_detect/
├── main.py                # 主流程：分割 + 检测 + 点云 + 贝塞尔曲线
├── demo_video.py          # 稀疏点云叠加 + 输出演示视频
├── method/                # 核心算法包
│   ├── method_pre.py      # 数据加载、掩码→点云映射、PLY 保存
│   └── method_algorithm.py# 贝塞尔/最小二乘/RANSAC 拟合、统计/DBSCAN 滤波
├── scripts/               # 实时相机相关脚本
│   ├── realsense_camera.py# 相机 + 分割 + 点云 + 曲线（实时）
│   ├── realsense_detect.py# 实时目标检测 + 三维坐标
│   ├── realsense_segment.py# 实时实例分割 + 点云
│   ├── camera.py          # 调用本机摄像头做预测
│   └── biaoding.py        # 相机内参测量
├── tests/                 # 测试脚本
│   ├── picture_curb.py    # 单张图：分割 + 边缘拟合
│   ├── two_test.py        # 检测 + 分割双模型推理耗时对比
│   ├── myseg_test.py      # 图片/视频分割测试
│   ├── draw_myseg.py      # matplotlib 可视化分割掩码
│   └── compare.py         # 预测点云与真值点云 IoU 评估
├── tools/                 # 数据录制 / 提取 / 查看工具
│   ├── record_bag.py      # 录制 RGB-D 到 .bag
│   ├── record_bag_imu.py  # 录制含 IMU 的 .bag
│   ├── extract_bag.py     # 从 .bag 提取帧
│   ├── repair_bag.py      # 修复 .bag 索引
│   ├── view_bag.py        # 查看 .bag 流信息
│   └── view_ply.py        # 用 Open3D 查看 .ply
├── train/                 # 训练 / 验证脚本
│   ├── train.py
│   └── val.py
├── weights/               # 模型权重
├── data/                  # 测试图片（before/ 为 labelme 标注样例）
└── docs/                  # 设计文档（架构 / 性能 / ZYNQ 部署）
```

## 模型权重

| 文件 | 说明 |
|------|------|
| `weights/best.pt` | 路沿实例分割模型（YOLOv8-seg 微调）|
| `weights/best1.pt` | 目标检测模型 |
| `weights/best_detect.pt` | 目标检测模型（另一版本）|
| `weights/dark.pt` | 弱光/夜间场景模型 |
| `weights/yolov8n.pt` | YOLOv8n 预训练（目标检测基础模型）|

> 训练数据的 `.bag` 录制文件与标注数据集体积较大（GB 级），未包含在本仓库；可用 `tools/record_bag.py` 自行录制，标注样例见 `data/before/`。

## 环境依赖

```bash
pip install -r requirements.txt
```

> - `pyrealsense2` 需要安装对应版本的 Intel RealSense SDK 2.0（librealsense）。
> - `torch` 建议按你的 CUDA 版本从 [pytorch.org](https://pytorch.org/get-started/locally/) 安装。
> - 程序默认从项目根目录运行（如 `python main.py`），模型与图片使用 `weights/`、`data/` 相对路径。

## 使用示例

```bash
# 主流程（处理 .bag 文件）
python main.py            # 修改文件内 bag_file_path 指向你的 .bag

# 单张图片分割 + 边缘拟合
python tests/picture_curb.py

# 双模型推理耗时对比
python tests/two_test.py

# 录制一段 .bag
python tools/record_bag.py
```

## 参考文档

- `docs/system_architecture.md` —— 系统总体架构设计
- `docs/performance_analysis.md` —— 性能瓶颈分析与优化
- `docs/zynq_implementation_design.md` —— ZYNQ7015 硬件加速部署方案