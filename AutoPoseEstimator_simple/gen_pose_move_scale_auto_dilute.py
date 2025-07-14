import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/user/pgp/CoorDiff')))
from T import *

# 设置中文字体和负号显示
font_name = "simhei"
plt.rcParams['font.family']= font_name # 指定字体，实际上相当于修改 matplotlibrc 文件　只不过这样做是暂时的　下次失效
plt.rcParams['axes.unicode_minus']=False # 正确显示负号，防止变成方框

import numpy as np
from scipy.spatial.transform import Rotation as R

import numpy as np
from scipy.spatial.transform import Rotation as R



def generate_diffused_trajectory(poses, x=0, y=0, z=0, fixed_ratio=0.5, start_rot_angle=0.0):
    """
    生成扩散后的轨迹数据，使得新轨迹的起点经过平移偏移，并且终点保持与原始轨迹终点一致。
    该版本仅对平移部分进行扩散和修正，旋转部分采取新的策略：
      - 末端 fixed_ratio 区域的点直接沿用原始轨迹的旋转；
      - 起点顺时针旋转给定角度 start_rot_angle（单位：度）；
      - 在起点（修改后的旋转）与固定区域首个点之间利用 Slerp 四元数插值进行旋转平滑拟合。
    
    参数:
      poses: (N, 4, 4) 的原始轨迹数据，每个元素为一个齐次变换矩阵
      x, y, z: 新起点相对于原始起点的平移偏移量
      fixed_ratio: 控制末端固定区域的比例（末端区域的旋转保持原始轨迹方向）
      start_rot_angle: 起点顺时针旋转的角度（单位：度）
      
    返回:
      new_poses: (N, 4, 4) 的扩散后轨迹数据，
                 轨迹点数与输入一致，
                 新轨迹起点经过偏移，终点与原始轨迹终点一致，
                 旋转部分前段经过 Slerp 平滑插值，末端区域保持原始旋转。
    """
    N = len(poses)
    if N < 2:
        raise ValueError("轨迹点数太少，不足以进行插值。")
    
    # ----- 平移部分处理 -----
    # 1. 提取原始平移部分
    orig_translations = np.array([T[:3, 3] for T in poses])
    
    # 定义原始起点和终点
    S_orig = orig_translations[0]       # 原始起点（完整3D）
    S_orig_end = orig_translations[-1]    # 原始终点（完整3D）
    
    # 为保持高度不变，只处理水平（x,y）部分：
    # 定义原始水平起点和终点
    S_orig_horiz = S_orig[:2]  
    S_orig_end_horiz = S_orig_end[:2]
    
    # 新起点水平坐标：将原始起点的水平部分加上偏移 [x,y]
    S_new_horiz = S_orig_horiz + np.array([x, y])
    # 注意：高度仍以原始起点的z值为基准，但后续每点的高度仍沿用原始数据
    
    # 归一化：计算每个点相对于原始水平起点的偏移(仅x,y)
    normalized_positions_horiz = np.array([T[:2] - S_orig_horiz for T in orig_translations])
    
    # 计算原始水平向量及其模长
    v_orig_horiz = S_orig_end_horiz - S_orig_horiz
    norm_v_orig_horiz = np.linalg.norm(v_orig_horiz)
    if norm_v_orig_horiz < 1e-6:
        raise ValueError("示范轨迹水平起点与终点距离太近，无法进行缩放。")
    
    # 计算新水平向量及其模长（由新起点到原始终点的水平距离）
    v_new_horiz = S_orig_end_horiz - S_new_horiz
    norm_v_new_horiz = np.linalg.norm(v_new_horiz)
    if norm_v_new_horiz < 1e-6:
        raise ValueError("新起点与原始终点水平距离过近，无法计算缩放。")
    
    # 计算水平缩放因子
    factor = norm_v_new_horiz / norm_v_orig_horiz
    
    # 计算水平旋转补偿矩阵 R_corr（关于z轴旋转，只影响 x,y）
    # 构造对应的3D向量，z分量置零
    v_orig_xy = np.array([v_orig_horiz[0], v_orig_horiz[1], 0])
    v_new_xy = np.array([v_new_horiz[0], v_new_horiz[1], 0])
    norm_v_orig_xy = np.linalg.norm(v_orig_xy)
    norm_v_new_xy = np.linalg.norm(v_new_xy)
    if norm_v_orig_xy < 1e-6 or norm_v_new_xy < 1e-6:
        R_corr = np.eye(3)
    else:
        theta_orig = np.arctan2(v_orig_xy[1], v_orig_xy[0])
        theta_new  = np.arctan2(v_new_xy[1], v_new_xy[0])
        delta_theta = theta_new - theta_orig
        R_corr = R.from_euler('z', delta_theta).as_matrix()
    
    # 生成新的平移部分：
    # 对于水平分量，采用 S_new_horiz + factor * (R_corr.dot(水平归一化偏移))
    # 对于垂直分量，则直接沿用原始高度
    new_positions = []
    for i, pos_horiz in enumerate(normalized_positions_horiz):
        # 先将水平部分扩展为三维向量（z置0），以便与 R_corr 相乘
        pos_3d = np.array([pos_horiz[0], pos_horiz[1], 0])
        new_horiz = S_new_horiz + factor * (R_corr.dot(pos_3d))[:2]
        # 高度（z）直接沿用原始轨迹对应点的高度
        new_z = orig_translations[i][2]
        new_positions.append(np.array([new_horiz[0], new_horiz[1], new_z]))
    new_positions = np.array(new_positions)

        # ----- 原始角度是否跟着旋转一起旋转 -----
    if args.need_rot:
        # 1. 提取原始旋转部分
        orig_rotations = [T[:3, :3] for T in poses]

        # 2. 创建旋转矩阵，绕终点的Z轴进行旋转
        theta = np.deg2rad(start_rot_angle)  # 顺时针旋转的角度（单位：弧度）

        # 3. 计算旋转矩阵
        R_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        # 4. 对每个轨迹点应用旋转
        new_rotations = []
        for i in range(N):
            # 这里直接将每个原始旋转矩阵与绕Z轴的旋转矩阵相乘，进行旋转
            rotated_matrix = R_z @ orig_rotations[i]  # 使用原始旋转矩阵进行旋转
            new_rotations.append(rotated_matrix)

        # ----- 组合平移与旋转构造新的齐次变换矩阵 -----
        new_poses = []
        for i in range(N):
            T_new = np.eye(4)
            T_new[:3, :3] = new_rotations[i]
            T_new[:3, 3] = new_positions[i]
            new_poses.append(T_new)
        new_positions = np.array(new_poses)



    # ----- 旋转部分处理 -----
    # 直接提取原始旋转部分作为基础
    if args.need_rot:
        # print(new_positions[0])
        orig_rotations = [T[:3, :3] for T in new_positions]
        orig_rotations[0] = poses[0][:3, :3]    # 确保第一个点旋转不变，方便旋转角度的统一
    else: 
        orig_rotations = [T[:3, :3] for T in poses]

    
    # ----- 旋转部分处理 -----
    # 直接提取原始旋转部分作为基础
    orig_rotations = [T[:3, :3] for T in poses]
    
    # 末端固定区域：计算固定区域点数
    n_fixed = max(1, int(fixed_ratio * N))
    interp_end = N - n_fixed  # 插值区域终点索引（固定区域的第一个点）
    
    # 起点旋转经过顺时针旋转 start_rot_angle 后作为插值起始点
    R_start_orig = R.from_matrix(orig_rotations[0])
    # 注意：start_rot_angle 参数单位为度，顺时针旋转对应负角度
    R_start_mod = R_start_orig * R.from_euler('x', -np.deg2rad(start_rot_angle))
    
    # 固定区域起始的旋转
    R_fixed_start = R.from_matrix(orig_rotations[interp_end])
    
    # 构造插值的时间参数，对插值区域内点进行 Slerp 插值
    if interp_end > 1:
        times = np.linspace(0, 1, interp_end)
    else:
        times = np.array([0])
    
    key_times = [0, 1]
    key_rots = R.from_quat([R_start_mod.as_quat(), R_fixed_start.as_quat()])
    slerp_obj = Slerp(key_times, key_rots)
    
    new_rotations = []
    # 对前 interp_end 个点做 Slerp 插值
    for i in range(interp_end):
        t_val = times[i]
        r_interp = slerp_obj(t_val)
        new_rotations.append(r_interp.as_matrix())
        
    # 后 n_fixed 个点直接采用原始旋转
    for i in range(interp_end, N):
        new_rotations.append(orig_rotations[i])
    
    # ----- 组合平移与旋转构造新的齐次变换矩阵 -----
    new_poses = []
    for i in range(N):
        T_new = np.eye(4)
        T_new[:3, :3] = new_rotations[i]
        # print(new_positions[i])
        if args.need_rot:
            T_new[:3, 3] = new_positions[i][:3, 3]
        else: 
            T_new[:3, 3] = new_positions[i]
        
        new_poses.append(T_new)
    new_poses = np.array(new_poses)
    
    return decimate_poses_uniformly_by_distance(new_poses, num_samples=num_samples, head_num_delete=head_num_delete, tail_num_delete=tail_num_delete)


def decimate_poses(poses, factor=2):
    """
    对轨迹数据进行稀释，即按照给定的采样步长对轨迹中的点进行取样，
    例如：factor=2 表示每隔 2 个点取一个，进而减小数据点数。
    
    参数:
      poses: (N, 4, 4) 轨迹数据，通常为齐次变换矩阵组成的数组.
      factor: 整数，表示采样步长（必须 >= 1）。
              当 factor=1 时，返回原数据；当 factor>1 时，返回稀释后的数据.
    
    返回:
      decimated_poses: 稀释后的轨迹数据，保证始终包含最后一点.
    """
    if factor <= 1:
        return poses
    decimated = poses[::factor]
    # 检查最后一个点是否为终点，不是则追加终点
    if not np.allclose(decimated[-1], poses[-1]):
        decimated = np.concatenate([decimated, poses[-1:]], axis=0)
    return decimated



def decimate_poses_uniformly_by_distance(poses, num_samples, head_num_delete=0, tail_num_delete=0):
    """
    对轨迹 poses 按几何距离进行均匀采样，采样后再删除开头的 head_num_delete 个点。
    
    参数:
      poses: (N, 4, 4) 的齐次变换矩阵数组，描述轨迹点
      num_samples: 采样时希望保留的轨迹点数（至少2个，包含首尾）
      head_num_delete: 从采样结果中要删除的最前面的点数（删除后至少剩1个结果点）
    
    返回:
      decimated_poses: 采样后的轨迹数据（经过删除处理），点的分布在轨迹距离上更均匀
    """
    if num_samples < 2:
        raise ValueError("num_samples 至少需要 2（起点和终点）")
    
    # 提取平移部分，计算欧氏距离
    positions = np.array([T[:3, 3] for T in poses])
    # 计算相邻点间距离
    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    # 计算累计距离数组
    cum_distance = np.concatenate([[0], np.cumsum(deltas)])
    
    # 均匀生成采样距离，从0到总长度
    total_length = cum_distance[-1]
    sample_dists = np.linspace(0, total_length, num_samples)
    
    # 根据采样距离获取对应的索引（采用最近的）
    new_indices = []
    j = 0
    for d in sample_dists:
        # 移动 j 直到 cum_distance[j+1] 与 d 的差值更小
        while j < len(cum_distance) - 1 and abs(cum_distance[j+1] - d) < abs(cum_distance[j] - d):
            j += 1
        new_indices.append(j)
    
    # 保证首尾一定包含
    new_indices[0] = 0
    new_indices[-1] = len(poses) - 1

    # 采样后的轨迹
    decimated_poses = poses[new_indices]
    
    # 检查删除数量是否合理
    if head_num_delete < 0 or head_num_delete >= len(decimated_poses):
        raise ValueError("head_num_delete 必须在 [0, len(decimated_poses)) 范围内")
    
    # 在采样结果中删除开头的 head_num_delete 个点
    decimated_poses = decimated_poses[head_num_delete: len(decimated_poses) - tail_num_delete]
    
    return decimated_poses

def process_offsets_single_output(offsets, input_folder, output_folder, obj_name, ref_name):
    """
    根据给定偏移量列表（例如：[[0.0, 1.0, 0.0], [0.1, 0.2, 0.0]]），对输入文件夹中的每个 episode
    生成扩散后的轨迹数据，并将原始数据和生成数据一起保存到同一输出文件夹中：
    
      - 原始数据保持原有的 episode_x 文件夹不变。
      - 生成数据按顺序累计编号，从原始 episode 数量之后开始命名。
      - 同时在输出文件夹中为每个偏移量生成一个 TXT 文件，其文件名对应该偏移坐标，
        文件内容记录使用的偏移信息。
    """
    # 获取所有 episode_x 文件夹（要求名称以 "episode_" 开头）
    episodes = sorted([ep for ep in os.listdir(input_folder) if ep.startswith("episode_")])
    num_original = len(episodes)
    print(f"在输入文件夹中共发现 {num_original} 个 episode。")
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 修改前复制文件夹部分，替换为处理原始数据后保存
    for ep in episodes:
        src_ep = os.path.join(input_folder, ep)
        dst_ep = os.path.join(output_folder, ep)
        os.makedirs(dst_ep, exist_ok=True)
        obj_poses_path = os.path.join(src_ep, f"{obj_name}_poses.npy")
        ref_poses_path = os.path.join(src_ep, f"{ref_name}_poses.npy")

        if os.path.exists(os.path.join(src_ep, f"hand_poses.npy")):
            shutil.copy(os.path.join(src_ep, f"hand_poses.npy"), os.path.join(dst_ep, f"hand_poses.npy"))
            print("成功复制T_o_hand_pose")
        
        # 检查必要文件是否存在
        if not os.path.exists(obj_poses_path) or not os.path.exists(ref_poses_path):
            print(f"Episode {ep} 缺少必要文件，跳过。")
            continue

        # 加载原始数据
        obj_poses = np.load(obj_poses_path)
        ref_poses = np.load(ref_poses_path)
        
        # 对原始数据进行均匀稀释，保证起点和终点一定包含
        decimated_obj_poses = decimate_poses_uniformly_by_distance(obj_poses, num_samples=num_samples, head_num_delete=head_num_delete, tail_num_delete=tail_num_delete)
        decimated_ref_poses = decimate_poses_uniformly_by_distance(ref_poses, num_samples=num_samples, head_num_delete=head_num_delete, tail_num_delete=tail_num_delete)
        
        # 保存处理后的数据到新文件夹中，文件名保持一致
        output_obj_path = os.path.join(dst_ep, f"{obj_name}_poses.npy")
        output_ref_path = os.path.join(dst_ep, f"{ref_name}_poses.npy")
        np.save(output_obj_path, decimated_obj_poses)
        np.save(output_ref_path, decimated_ref_poses)
        
        print(f"处理并保存原始数据 {ep} 完成，现有长度{len(decimated_obj_poses)}")

    # # 先将原始数据复制到输出文件夹中（保持原有 episode_x 文件夹名称）
    # for ep in episodes:
    #     src_ep = os.path.join(input_folder, ep)
    #     dst_ep = os.path.join(output_folder, ep)
    #     if os.path.exists(dst_ep):
    #         shutil.rmtree(dst_ep)
    #     shutil.copytree(src_ep, dst_ep)
    #     print(f"复制原始 {ep} 完成。")
    
    # 固定的 T_b_cam（根据实际情况设定）
    # T_b_cam = np.array([
    #     [-0.0617699,  0.55345311, -0.83058662, 1.02605127],
    #     [ 0.99730353,  0.00118691, -0.07337758, 0.00620415],
    #     [-0.03962522, -0.83287949, -0.55203405, 0.3469516],
    #     [ 0.,         0.,         0.,         1.        ]
    # ])
    T_b_cam = T_b_cam_init
    
    # 新生成的 episode 编号从原始数目开始累加
    current_index = num_original
    
    if offsets is not None:
        # 对于每个偏移量（形式为 [x, y, z]），生成对应的 txt 文件和新数据
        for offset in offsets:
            x, y, z , angle= offset
            
            # 对每个原始 episode 生成扩散数据
            for ep in episodes:
                ep_input_path = os.path.join(input_folder, ep)
                obj_poses_path = os.path.join(ep_input_path, f"{obj_name}_poses.npy")
                ref_poses_path = os.path.join(ep_input_path, f"{ref_name}_poses.npy")
                
                # 检查必要文件是否存在
                if not os.path.exists(obj_poses_path) or not os.path.exists(ref_poses_path):
                    print(f"Episode {ep} 缺少必要文件，跳过。")
                    continue
                
                # 加载并转换数据
                T_cam_o_poses = np.load(obj_poses_path)
                T_b_o_poses = [T_b_cam @ T_cam_o for T_cam_o in T_cam_o_poses]
                
                # 进行数据生成
                diffused_T_b_o_poses = generate_diffused_trajectory(T_b_o_poses, x=x, y=y, z=z, start_rot_angle=angle)
                # 最终保存的需要的是相机坐标系
                diffused_T_cam_o_poses = [np.linalg.inv(T_b_cam) @ T_b_o for T_b_o in diffused_T_b_o_poses]
                diffused_T_cam_o_poses = np.array(diffused_T_cam_o_poses)
                
                # 新 episode 文件夹名：episode_{current_index}
                new_ep_folder = os.path.join(output_folder, f"episode_{current_index}")
                os.makedirs(new_ep_folder, exist_ok=True)
                # 保存生成的对象位姿数据
                output_obj_path = os.path.join(new_ep_folder, f"{obj_name}_poses.npy")
                np.save(output_obj_path, diffused_T_cam_o_poses)
                # 参照数据直接拷贝过去
                # output_ref_path = os.path.join(new_ep_folder, f"{ref_name}_poses.npy")
                # shutil.copy(ref_poses_path, output_ref_path)
                # 参照数据经稀释后再过去
                ref_poses = np.load(ref_poses_path)
                decimated_ref_poses = decimate_poses_uniformly_by_distance(ref_poses, num_samples=num_samples, head_num_delete=head_num_delete, tail_num_delete=tail_num_delete)
                output_ref_path = os.path.join(new_ep_folder, f"{ref_name}_poses.npy")
                np.save(output_ref_path, decimated_ref_poses)
                # 在输出文件夹内生成 TXT 文件，文件名对应偏移坐标
                txt_filename = f"offset_x{x}_y{y}_z{z}_base_{ep}.txt"
                txt_filepath = os.path.join(new_ep_folder, txt_filename)
                with open(txt_filepath, 'w', encoding='utf-8') as f_txt:
                    f_txt.write(f"生成数据采用的偏移坐标为: [{x}, {y}, {z}]\n")
                # print(f"生成偏移信息文件: {txt_filename}")
                
                print(f"生成 {new_ep_folder} （基于 {ep} 应用偏移 [{x}, {y}, {z}]）完成。")
                
                current_index += 1





def add_poses(ax, poses, label_prefix):
    """
    遍历给定轨迹的每个姿态，将 T_cam_o 通过 T_b_cam 转换到基座坐标系下，
    然后绘制原点以及坐标轴方向（x: 蓝色, y: 绿色, z: 红色）。
    label_prefix 用于区分不同轨迹的标签。
    """

    # 设定每个坐标轴箭头的缩放因子
    scale = 0.02


    # 获取已有图例标签，便于只在第一次添加标签
    existing_labels = ax.get_legend_handles_labels()[1]
    for T_b_o in poses:
        # 计算物体（轨迹）在基座坐标系下的变换矩阵
        origin = T_b_o[:3, 3]
        # 计算坐标轴方向
        x_axis = T_b_o[:3, 0] * scale 
        y_axis = T_b_o[:3, 1] * scale
        z_axis = T_b_o[:3, 2] * scale
        
        # 绘制原点
        ax.scatter(origin[0], origin[1], origin[2], color='k', s=20)
        
        # 绘制坐标轴箭头，并仅为第一次出现添加标签
        if label_prefix + ' X' not in existing_labels:
            ax.quiver(origin[0], origin[1], origin[2],
                      x_axis[0], x_axis[1], x_axis[2],
                      color='b', label=label_prefix + ' X')
        else:
            ax.quiver(origin[0], origin[1], origin[2],
                      x_axis[0], x_axis[1], x_axis[2],
                      color='b')
            
        if label_prefix + ' Y' not in existing_labels:
            ax.quiver(origin[0], origin[1], origin[2],
                      y_axis[0], y_axis[1], y_axis[2],
                      color='g', label=label_prefix + ' Y')
        else:
            ax.quiver(origin[0], origin[1], origin[2],
                      y_axis[0], y_axis[1], y_axis[2],
                      color='g')
            
        if label_prefix + ' Z' not in existing_labels:
            ax.quiver(origin[0], origin[1], origin[2],
                      z_axis[0], z_axis[1], z_axis[2],
                      color='r', label=label_prefix + ' Z')
        else:
            ax.quiver(origin[0], origin[1], origin[2],
                      z_axis[0], z_axis[1], z_axis[2],
                      color='r')
        # 更新标签集合
        existing_labels = ax.get_legend_handles_labels()[1]


def visualize_T_b_o(poses1, poses2):
    # 创建一个三维图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制相机坐标系（在基座坐标系下）
    cam_origin = T_b_cam[:3, 3]
    ax.scatter(cam_origin[0], cam_origin[1], cam_origin[2], color='k', s=20, label='相机')
    cam_x_axis = T_b_cam[:3, 0] * 0.1
    cam_y_axis = T_b_cam[:3, 1] * 0.1
    cam_z_axis = T_b_cam[:3, 2] * 0.1
    ax.quiver(cam_origin[0], cam_origin[1], cam_origin[2],
            cam_x_axis[0], cam_x_axis[1], cam_x_axis[2], color='b')
    ax.quiver(cam_origin[0], cam_origin[1], cam_origin[2],
            cam_y_axis[0], cam_y_axis[1], cam_y_axis[2], color='g')
    ax.quiver(cam_origin[0], cam_origin[1], cam_origin[2],
            cam_z_axis[0], cam_z_axis[1], cam_z_axis[2], color='r')
    
    # 添加轨迹两个轨迹
    add_poses(ax, poses1, '轨迹1')
    add_poses(ax, poses2, '轨迹2')

    # # 添加图例
    ax.legend()

    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('两个物体轨迹在基座坐标系下的姿态')

    # 设置 3D 坐标轴的比例为1:1:1，确保各轴单位一致
    ax.set_xlim([-0.5, 1])
    ax.set_ylim([-0.5, 1])
    ax.set_zlim([-0.5, 1])


    plt.show()


def generate_noisy_trajectory(T_b_o_poses, noise_std_translation=0.001, noise_std_rotation=0):
    """
    生成新的轨迹数据，其中起点和终点与原始数据一致，
    中间轨迹点的平移和旋转部分分别添加高斯噪声。

    参数:
      T_b_o_poses: (N, 4, 4) 的原始轨迹数据，每个元素为一个齐次变换矩阵
      noise_std_translation: 平移部分高斯噪声的标准差（与平移单位一致）
      noise_std_rotation: 旋转部分噪声的标准差（以弧度为单位，在旋转向量空间添加噪声）

    返回:
      new_poses: (N, 4, 4) 的新轨迹数据
    """
    from scipy.spatial.transform import Rotation as R
    T_b_o_poses = np.array(T_b_o_poses)
    N = T_b_o_poses.shape[0]
    new_poses = []
    
    for i in range(N):
        # 起点和终点保持不变
        if i == 0 or i == N - 1:
            new_poses.append(T_b_o_poses[i])
        else:
            T = T_b_o_poses[i]
            # 提取原始平移部分，并加上高斯噪声
            t = T[:3, 3]
            noisy_t = t + np.random.normal(0, noise_std_translation, size=3)
            
            # 提取原始旋转部分，加上高斯噪声
            R_orig = T[:3, :3]
            r_obj = R.from_matrix(R_orig)
            rotvec = r_obj.as_rotvec()
            noisy_rotvec = rotvec + np.random.normal(0, noise_std_rotation, size=3)
            new_R = R.from_rotvec(noisy_rotvec).as_matrix()
            
            # 重新组合为齐次变换矩阵
            T_new = np.eye(4)
            T_new[:3, :3] = new_R
            T_new[:3, 3] = noisy_t
            new_poses.append(T_new)
    
    return np.array(new_poses)


# 使用示例：
if __name__ == '__main__':
    from offsets_all import *
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--obj_name", required=True)
    parser.add_argument("--ref_name", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--num_samples", required=True, type=int)
    parser.add_argument("--output_name", required=True, type=str)
    parser.add_argument("--need_rot", action='store_true', default=False)
    args = parser.parse_args()

     # 任务和文件名设置
    task = args.task
    obj_name = args.obj_name
    ref_name = args.ref_name
    # 输入文件夹路径，假定结构与原始代码一致，如：output/data_real/put_new6
    input_folder = f"/home/user/pgp/CoorDiff/coordiff_real_world/final_data/{task}"
    # 输出文件夹路径（全部数据都存储在此文件夹中）
    output_folder = f"/home/user/pgp/CoorDiff/coordiff_real_world/final_data/{args.output_name}"
    
    # 示例偏移量列表，形式为 [[x, y, z], ...]
    """
    y = 0.3为对称过去
    x = -0.242为到黑线对面
    """
    # dilute_factor = 5
    num_samples = args.num_samples
    head_num_delete = 0
    tail_num_delete = 0
    offsets = None

    # offsets = [[-0.242, 0.0, 0.0, 0.0], [-0.242, 0.15, 0.0, 0.0], [-0.242, 0.3, 0.0, 0.0], [0.0, 0.3, 0.0, 0.0]]
    # offsets = [[0, 0.0, 0.0, 90]]
    # offsets = [[-0.242, 0.3, 0.0]]
    # offsets = [[-0.242, 0.15, 0.0]]
    # offsets = [
#     [0.0, 0.0, 0.0, 90], [0.0, 0.0, 0.0, 180], [0.0, 0.0, 0.0, 270],
#     [-0.242, 0.0, 0.0, 0], [-0.242, 0.0, 0.0, 90], [-0.242, 0.0, 0.0, 180], [-0.242, 0.0, 0.0, 270],
#     [-0.242, 0.15, 0.0, 0], [-0.242, 0.15, 0.0, 90], [-0.242, 0.15, 0.0, 180], [-0.242, 0.15, 0.0, 270],
#     [-0.242, 0.3, 0.0, 0], [-0.242, 0.3, 0.0, 90], [-0.242, 0.3, 0.0, 180], [-0.242, 0.3, 0.0, 270], 
#     [0.0, 0.3, 0.0, 0], [0.0, 0.3, 0.0, 90], [0.0, 0.3, 0.0, 180], [0.0, 0.3, 0.0, 270]
#     ]

    # teapot扩散
    if args.obj_name == "teapot2":
        if args.model == '15dots':
            offsets = teapot_15dots_30o
        if args.model == '50dots':
            offsets = teapot_50dots_30o
        elif args.model == 'all':
            offsets = teapot_all_30o
    
    elif args.obj_name == "bluemug":
        if args.model == '15dots':
            offsets = mug_15dots_30o
        if args.model == '50dots':
            offsets = mug_50dots_30o
        elif args.model == 'all':
            offsets = mug_all_30o
        elif args.model == 'cover':
            offsets = mug_cover_30o

    elif args.obj_name == "pinkmug":
        if args.model == '15dots':
            offsets = holder_15dots_30o
        if args.model == '50dots':
            offsets = holder_50dots_30o
        elif args.model == 'all':
            offsets = holder_all_30o

    elif args.obj_name == "steve":
        if args.model == '15dots':
            offsets = steve_15dots_30o
        if args.model == '50dots':
            offsets = steve_50dots_30o
        elif args.model == 'all':
            offsets = steve_all_30o

    else:
        offsets = []


        # offsets = [
        #     [0.0, 0.0, 0.0, 90], [0.0, 0.0, 0.0, 180], [0.0, 0.0, 0.0, 270]
        #     ,[0.0, 0.15, 0.0, 0], [0.0, 0.15, 0.0, 90], [0.0, 0.15, 0.0, 180], [0.0, 0.15, 0.0, 270]
        #     ,[-0.27, 0.15, 0.0, 0], [-0.27, 0.15, 0.0, 90], [-0.27, 0.15, 0.0, 180], [-0.27, 0.15, 0.0, 270]
        #     ,[-0.27, 0, 0.0, 0], [-0.27, 0, 0.0, 90], [-0.27, 0, 0.0, 180], [-0.27, 0, 0.0, 270]
        #     ,[-0.27, -0.15, 0.0, 0], [-0.27, -0.15, 0.0, 90], [-0.27, -0.15, 0.0, 180], [-0.27, -0.15, 0.0, 270]
        #     ]

    # mug扩散
    # offsets = [
    #             [0.0, 0.0, 0.0, 90], [0.0, 0.0, 0.0, 180], [0.0, 0.0, 0.0, 270]
    #            ,[0.0, 0.15, 0.0, 0], [0.0, 0.15, 0.0, 90], [0.0, 0.15, 0.0, 180], [0.0, 0.15, 0.0, 270]
    #            ,[-0.27, 0.15, 0.0, 0], [-0.27, 0.15, 0.0, 90], [-0.27, 0.15, 0.0, 180], [-0.27, 0.15, 0.0, 270]
    #            ,[-0.27, 0, 0.0, 0], [-0.27, 0, 0.0, 90], [-0.27, 0, 0.0, 180], [-0.27, 0, 0.0, 270]
    #            ,[-0.27, -0.15, 0.0, 0], [-0.27, -0.15, 0.0, 90], [-0.27, -0.15, 0.0, 180], [-0.27, -0.15, 0.0, 270]
    #            ]

    # offsets = [
    #             [-0.27, 0, 0.0, 0]
    #             # [0.0, 0.0, 0.0, 90], [0.0, 0.0, 0.0, 180], [0.0, 0.0, 0.0, 270]
    #         #    ,[0.0, 0.15, 0.0, 0], [0.0, 0.15, 0.0, 90], [0.0, 0.15, 0.0, 180], [0.0, 0.15, 0.0, 270]
    #         #    ,[-0.27, 0.15, 0.0, 0], [-0.27, 0.15, 0.0, 90], [-0.27, 0.15, 0.0, 180], [-0.27, 0.15, 0.0, 270]
    #         #    ,[-0.27, 0, 0.0, 0], [-0.27, 0, 0.0, 90], [-0.27, 0, 0.0, 180], [-0.27, 0, 0.0, 270]
    #         #    ,[-0.27, -0.15, 0.0, 0], [-0.27, -0.15, 0.0, 90], [-0.27, -0.15, 0.0, 180], [-0.27, -0.15, 0.0, 270]
    #            ]

    
    process_offsets_single_output(offsets, input_folder, output_folder, obj_name, ref_name)
    
    
    """
    测试用例程
    """
    # task = "put_new6"
    # obj_name = "teapot2"
    # ref_name = "coastercat"
    # # task_path = f"output/data_real/{task}_gen_nomove"
    # task_path = f"output/data_real/{task}"
    # offset = 1


    # entries = os.listdir(task_path)
    # length = len(entries)
    # print(f"原始数据共有{length}条")


    # for i in range(length):
    #     # 新建复制目录
    #     copy_path = f"output/data_real/{task}_gen_scale/episode_{i}"
    #     if not os.path.exists(copy_path):
    #         os.makedirs(copy_path)
    #     # 新建输出目录
    #     output_path = f"output/data_real/{task}_gen_scale/episode_{i+length * offset}"
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)
        
    #     # 目标路径处理
    #     obj_poses_path = os.path.join(task_path, f'episode_{i}/{obj_name}_poses.npy')
    #     ref_poses_path = os.path.join(task_path, f'episode_{i}/{ref_name}_poses.npy')
    #     output_obj_poses_path = os.path.join(output_path, f'{obj_name}_poses.npy')
    #     output_ref_poses_path = os.path.join(output_path, f'{ref_name}_poses.npy')


    #     T_cam_o_poses = np.load(os.path.join(obj_poses_path))

    #     T_b_cam = np.array([
    #         [-0.0617699,   0.55345311, -0.83058662,  1.02605127],
    #         [ 0.99730353,  0.00118691, -0.07337758,  0.00620415],
    #         [-0.03962522, -0.83287949, -0.55203405,  0.3469516 ],
    #         [ 0.        ,  0.        ,  0.        ,  1.        ]
    #     ])

    #     T_b_o_poses = []
    #     for T_cam_o in T_cam_o_poses:
    #         T_b_o_poses.append(T_b_cam @ T_cam_o)    # 获取物体在机械臂基座下的坐标
        
    #     diffused_T_b_o_poses = generate_diffused_trajectory(T_b_o_poses,  y=0.3)

    #     # diffused_T_b_o_poses = generate_diffused_trajectory(T_b_o_poses)
        
    #     # diffused_T_b_o_poses = generate_noisy_trajectory(T_b_o_poses)
    #     # visualize_T_b_o(T_b_o_poses, diffused_T_b_o_poses)

    #     # 保存扩散后的轨迹数据（可选）给他取逆
    #     diffused_T_cam_o_poses = []
    #     for diffused_T_b_o in diffused_T_b_o_poses:
    #         diffused_T_cam_o_poses.append(np.linalg.inv(T_b_cam) @ diffused_T_b_o)
        
    #     # visualize_T_b_o(T_cam_o_poses ,diffused_T_cam_o_poses)

    #     # 保存生成数据到新文件夹下
    #     np.save(output_obj_poses_path, diffused_T_cam_o_poses)
    #     shutil.copy(ref_poses_path, output_ref_poses_path)
    #     print(f"原轨迹点数{len(T_b_o_poses)}")
    #     print(f"生成轨迹点数{len(diffused_T_b_o_poses)}")

                
    #     # 保存原始数据到新文件夹中
    #     shutil.copy(os.path.join(task_path, f'episode_{i}/{obj_name}_poses.npy'), os.path.join(copy_path, f'{obj_name}_poses.npy'))
    #     shutil.copy(os.path.join(task_path, f'episode_{i}/{ref_name}_poses.npy'), os.path.join(copy_path, f'{ref_name}_poses.npy'))


