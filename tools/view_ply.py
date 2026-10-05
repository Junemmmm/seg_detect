import open3d as o3d

def view_ply_file(filename):
    # 读取 .ply 文件
    pcd = o3d.io.read_point_cloud(filename)

    # 显示点云
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    view_ply_file("maskPointCloud262_mask_0.ply")


# import open3d as o3d
# import numpy as np
#
# # 读取 PLY 文件并提取点云数据
# point_cloud = []
# with open("output_maskPointCloud_frame32_mask_0.ply", "r") as file:
#     # 跳过头部行
#     for line in file:
#         if line.startswith("end_header"):
#             break
#
#     # 处理点云数据
#     for line in file:
#         # 清理每一行，去掉空格和括号
#         cleaned_line = line.strip().replace('(', '').replace(')', '').replace(',', '')
#
#         # 确保有数据可供分割
#         if cleaned_line:
#             # 将字符串分割为列表
#             values = cleaned_line.split()
#             # 每三个值组成一个点
#             try:
#                 for i in range(0, len(values), 3):
#                     # 确保有足够的值组成一个点
#                     if i + 2 < len(values):
#                         x = float(values[i])
#                         y = float(values[i + 1])
#                         z = float(values[i + 2])
#                         point_cloud.append((x, y, z))
#             except ValueError as e:
#                 print(f"Error processing line: {cleaned_line} -> {e}")
#
# # 创建 Open3D 点云对象
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud)
#
# # 显示点云
# o3d.visualization.draw_geometries([pcd])
