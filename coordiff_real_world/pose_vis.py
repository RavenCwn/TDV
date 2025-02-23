import numpy as np
import plotly.graph_objects as go
import os
import re  # 用于提取文件名中的数字


def get_numeric_key(file_name):
    """
    提取文件名中的数字部分以供排序使用
    """
    match = re.search(r'(\d+)', file_name)  # 提取连续的数字
    return int(match.group()) if match else 0


def load_pose_from_txt(file_path):
    """
    从指定文件加载4x4变换矩阵
    Args:
        file_path: .txt 文件路径
    Returns:
        pose: 变换矩阵（numpy array, shape 4x4）
    """
    pose = np.loadtxt(file_path, delimiter=' ')
    if pose.shape == (4, 4):
        return pose
    else:
        raise ValueError(f"文件 {file_path} 中的矩阵不是有效的 4x4 变换矩阵！")


def plotly_visualize_transformations_from_folder(folder_path):
    """
    使用 Plotly 可交互可视化工具可视化变换矩阵，并将中心点依次连线

    Args:
        folder_path: 包含 .txt 文件的文件夹路径
    """
    # 获取所有符合 pose_x.txt 格式的文件路径
    txt_files = [f for f in os.listdir(folder_path) if f.startswith('pose_') and f.endswith('.txt')]

    # 按照文件名中的数字部分排序
    txt_files = sorted(txt_files, key=get_numeric_key)

    if not txt_files:
        print(f"指定的文件夹 {folder_path} 中没有找到符合条件的 pose_x.txt 文件！")
        return

    poses = []
    specific_points = {}  # 用于存储指定点的位置信息

    # 按照排序后的文件名递增顺序加载所有变换矩阵
    for i, txt_file in enumerate(txt_files):
        file_path = os.path.join(folder_path, txt_file)
        print(f"加载变换矩阵文件: {file_path}")
        poses.append(load_pose_from_txt(file_path))

    # 创建绘图对象
    fig = go.Figure()

    # 初始化轨迹序列
    trajectory_x, trajectory_y, trajectory_z = [], [], []

    # 遍历每一个矩阵，绘制中心点及局部坐标系
    for i, pose in enumerate(poses):
        origin = pose[:3, 3]  # 提取中心点 (x, y, z)
        x_axis = pose[:3, 0] * 0.2  # 缩放箭头长度
        y_axis = pose[:3, 1] * 0.2
        z_axis = pose[:3, 2] * 0.2

        # 添加中心点到轨迹
        trajectory_x.append(origin[0])
        trajectory_y.append(origin[1])
        trajectory_z.append(origin[2])

        # 添加中心点
        fig.add_trace(go.Scatter3d(
            x=[origin[0]], y=[origin[1]], z=[origin[2]],
            mode='markers',
            marker=dict(size=5, color='black'),
            name=f"Pose {i + 1} Center"
        ))

        # 打印指定点的位置信息
        if i == dot1 or i == dot2:  # 打印第一个点和第143个点
            print(f"第 {i} 点：x = {origin[0]:.4f}, y = {origin[1]:.4f}, z = {origin[2]:.4f}")
            specific_points[i] = origin  # 保存特定点的位置信息

        # 添加 X 轴
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + x_axis[0]],
            y=[origin[1], origin[1] + x_axis[1]],
            z=[origin[2], origin[2] + x_axis[2]],
            mode='lines',
            line=dict(color='red', width=5),
            name=f"Pose {i + 1} X"
        ))

        # 添加 Y 轴
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + y_axis[0]],
            y=[origin[1], origin[1] + y_axis[1]],
            z=[origin[2], origin[2] + y_axis[2]],
            mode='lines',
            line=dict(color='green', width=5),
            name=f"Pose {i + 1} Y"
        ))

        # 添加 Z 轴
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + z_axis[0]],
            y=[origin[1], origin[1] + z_axis[1]],
            z=[origin[2], origin[2] + z_axis[2]],
            mode='lines',
            line=dict(color='blue', width=5),
            name=f"Pose {i + 1} Z"
        ))


    # 打印并计算指定两点的距离
    if dot1 in specific_points and dot2 in specific_points:
        p1 = specific_points[dot1]
        p2 = specific_points[dot2]
        distance = np.linalg.norm(p2 - p1)
        print(f"第{dot1}点和第{dot2}点之间的距离为: {distance:.4f}m")

    # 添加轨迹连线
    fig.add_trace(go.Scatter3d(
        x=trajectory_x,
        y=trajectory_y,
        z=trajectory_z,
        mode='lines',
        line=dict(color='black', width=3),
        name="Trajectory"
    ))

    # 设置图形布局
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[-1, 1]),
            yaxis=dict(nticks=10, range=[-1, 1]),
            zaxis=dict(nticks=10, range=[-1, 1]),
            aspectmode='data'
        ),
        title="Interactive 3D Visualizations of Transformations with Trajectory",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 显示交互式图形
    fig.show()


# 调用可视化函数，指定文件夹路径
dot1 = 0
dot2 = 325
folder_path = "../episode_0/track_pose_cup1"  # 更改为你的文件夹路径
plotly_visualize_transformations_from_folder(folder_path)
