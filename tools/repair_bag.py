import pyrealsense2 as rs

def repair_bag_file(file_path):
    try:
        rs.config().enable_device_from_file(file_path, repeat_playback=False)
        print(f"索引修复成功: {file_path}")
    except RuntimeError as e:
        print(f"索引修复失败: {e}")

# 调用修复函数
repair_bag_file("../output10.bag")
