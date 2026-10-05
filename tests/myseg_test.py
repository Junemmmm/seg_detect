import cv2
from ultralytics import YOLO
import sys
import os
import time


# test iamge
# 保存图片需要按下 Q 键
def test_img():
    # 记录开始时间
    start_time = time.time()

	# 训练好的模型权重路径
    model = YOLO("weights/best.pt")

    # 记录模型加载时间
    load_time = time.time()

    # 读取图片的路径
    img = cv2.imread("data/shelter2.png")

    # # 自定义输入图像的分辨率
    # custom_width = 640  # 修改为所需宽度
    # custom_height = 480  # 修改为所需高度
    # img_resized = cv2.resize(img, (custom_width, custom_height))  # 调整分辨率

    # 对图片进行推理
    res = model(img)

    # 记录推理结束时间
    inference_time = time.time()

    # 获取标注后的图片
    ann = res[0].plot()

    # 自定义显示分辨率（宽 x 高）
    display_width = 640  # 修改为显示图片的宽度
    display_height = 480  # 修改为显示图片的高度
    ann_resized = cv2.resize(ann, (display_width, display_height))  # 调整显示分辨率

    # 记录绘制标注时间
    plot_time = time.time()

    # 显示标注后的图片
    while True:
        cv2.imshow("yolo", ann_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # 获取当前脚本的目录
    cur_path = sys.path[0]
    print(cur_path, sys.path)

    output_dir = os.path.join(cur_path, 'data')  # 保存图片的输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 如果目录不存在则创建

    output_file = os.path.join(output_dir, 'out.jpg')  # 设置完整的文件路径

    # 保存处理后的图像
    cv2.imwrite(output_file, ann)
    print(f"Image saved at: {output_file}")

    # 输出时间统计
    print(f"Model loading time: {load_time - start_time:.2f} seconds")
    print(f"Inference time: {inference_time - load_time:.2f} seconds")
    print(f"Annotation drawing time: {plot_time - inference_time:.2f} seconds")



# test video
# 保存视频需要运行完整个视频，按 Q 键结束 （在保存视频的时候有点BUG）
def test_video():
    # 写入训练好的模型权重路径
    model = YOLO("weights/best.pt")
    # 测试视频存放目录
    pa = "seg_train/test/test_video.mp4"
    cap = cv2.VideoCapture(pa)
    # 调用设备自身摄像头
    # cap = cv2.VideoCapture(0) # -1q
    # 设置视频尺寸
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (w, h)
    # 第一个参数是将检测视频存储的路径, 输出保存的视频的帧率需要设输入的帧率保持一致
    video_writer = cv2.VideoWriter('seg_train/test/save.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,size)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            res = model(frame)
            ann = res[0].plot()

            # 确保图像尺寸匹配
            ann_resized = cv2.resize(ann, size)

            cv2.imshow("yolo", ann_resized)
            video_writer.write(ann_resized)
            key = cv2.waitKey(1)  # 等待用户输入
            if key & 0xFF == ord('q') or key == 27:
                break


    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()



test_img()
# test_video()
