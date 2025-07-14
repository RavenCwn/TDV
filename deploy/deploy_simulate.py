import numpy as np
import argparse
import os
from PIL import Image

import hydra
import torch
from easydict import EasyDict
from transformers import AutoTokenizer, AutoModel
from coordiff.utils.transform import *
from coordiff.models import *
from hydra import initialize, compose
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm
import time
import shutil
from eval_utils import *
import urx
from urx.gripper import Robotiq_Two_Finger_Gripper
import cv2
import lcm
import threading
from lcm_type.Matrix4x4Flat import Matrix4x4Flat
from T import T_b_cam_init as T_w_cam_init
import math
lc = lcm.LCM()

DEBUG = 1 # 0：机器人不动；1：机器人动；2: 复位；

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib
# matplotlib.use('qt5agg')  # 或者 'tkagg'
# plt.ion()  # 开启交互模式


# 设置中文字体和负号显示
font_name = "simhei"
plt.rcParams['font.family']= font_name # 指定字体，实际上相当于修改 matplotlibrc 文件　只不过这样做是暂时的　下次失效
plt.rcParams['axes.unicode_minus']=False # 正确显示负号，防止变成方框





def add_poses(ax, T_b_cam, poses, label_prefix, scale = 0.02):
    """
    遍历给定轨迹的每个姿态，将 T_cam_o 通过 T_b_cam 转换到基座坐标系下，
    然后绘制原点以及坐标轴方向（x: 蓝色, y: 绿色, z: 红色）。
    label_prefix 用于区分不同轨迹的标签。
    """

    # 设定每个坐标轴箭头的缩放因子
    # scale = 0


    # 获取已有图例标签，便于只在第一次添加标签
    existing_labels = ax.get_legend_handles_labels()[1]
    for i, T_cam_o in enumerate(poses):
        # 计算物体（轨迹）在基座坐标系下的变换矩阵
        T_b_o = T_b_cam @ T_cam_o
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
        
        
        # 在每条轨迹的起点处做特别标记
        if i == 0:  # 在第一个轨迹（或任何你希望重点标记的轨迹）时
            ax.scatter(origin[0], origin[1], origin[2], color='magenta', s=100, label=label_prefix + ' 起点')

        # 更新标签集合
        existing_labels = ax.get_legend_handles_labels()[1]


def visualize_T_b_o(generate_pose, raw_poses):
    # 创建一个三维图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    T_b_cam = np.array([
[ 0.00824559,  0.55120538, -0.83432886,  1.02815004],
 [ 0.99795435, -0.05743326, -0.02808102, -0.00693304],
 [-0.06339664, -0.83239056, -0.55055137,  0.39449762],
 [ 0.        ,  0.        ,  0.        ,  1.        ],
    ])

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
    
    # 绘制各轨迹的坐标系（原点和坐标轴）
    add_poses(ax, T_b_cam, generate_pose, scale=0, label_prefix='生成轨迹')

    for raw_pose in raw_poses:
        if raw_pose is None:
            continue
        add_poses(ax, T_b_cam, raw_pose, scale=0.01, label_prefix='原始数据')
    
    # 提取每个轨迹在基座坐标系下的所有平移点（原点），并用直线连接
    def get_origins(poses, T_b_cam):
        origins = []
        for T_cam_o in poses:
            T_b_o = T_b_cam @ T_cam_o
            origins.append(T_b_o[:3, 3])
        return np.array(origins)
    
    origins1 = get_origins(generate_pose, T_b_cam)
    ax.plot(origins1[:, 0], origins1[:, 1], origins1[:, 2], color='blue', label='轨迹1 连接线')

    for raw_pose in raw_poses:
        origins2 = get_origins(raw_pose, T_b_cam)
        ax.plot(origins2[:, 0], origins2[:, 1], origins2[:, 2], color='orange', label='轨迹2 连接线')
    
    # 添加图例
    # ax.legend()

    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('两个物体轨迹在基座坐标系下的姿态')
    
    plt.gca().set_aspect('equal')
    plt.show()


def visualize_generate_pose(generate_pose):
    # 创建一个三维图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    T_b_cam = np.array([
[ 0.00824559,  0.55120538, -0.83432886,  1.02815004],
 [ 0.99795435, -0.05743326, -0.02808102, -0.00693304],
 [-0.06339664, -0.83239056, -0.55055137,  0.39449762],
 [ 0.        ,  0.        ,  0.        ,  1.        ],
    ])

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
    
    # 绘制各轨迹的坐标系（原点和坐标轴）
    add_poses(ax, T_b_cam, generate_pose, scale=0, label_prefix='生成轨迹')
    
    # 提取每个轨迹在基座坐标系下的所有平移点（原点），并用直线连接
    def get_origins(poses, T_b_cam):
        origins = []
        for T_cam_o in poses:
            T_b_o = T_b_cam @ T_cam_o
            origins.append(T_b_o[:3, 3])
        return np.array(origins)
    
    origins1 = get_origins(generate_pose, T_b_cam)
    ax.plot(origins1[:, 0], origins1[:, 1], origins1[:, 2], color='blue', label='轨迹1 连接线')
    
    # 添加图例
    # ax.legend()

    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('两个物体轨迹在基座坐标系下的姿态')
    
    plt.gca().set_aspect('equal')
    plt.show()


# 处理接收到的矩阵消息
def matrix_handler(channel, data):
    global relevant_pose_T
    msg = Matrix4x4Flat.decode(data)
    
    # 将一维数组重新转换为4x4矩阵
    matrix = np.array(msg.matrix).reshape(4, 4)
    relevant_pose_T = matrix
    # print("Received matrix:")
    # print(matrix)

# 订阅MATRIX_CHANNEL频道
subscription = lc.subscribe("relevant_pose", matrix_handler)
subscription.set_queue_capacity(1)

# 处理LCM消息的函数
def handle_lcm():
    print("Waiting for messages...")
    while True:
        lc.handle()  # 这是阻塞调用，会一直等待消息


# 创建一个线程来运行handle_lcm函数
handle_thread = threading.Thread(target=handle_lcm)
handle_thread.daemon = True  # 设置为守护线程，程序退出时会自动退出
handle_thread.start()


def T_to_6D(T, rot_type):
    pos = T[:3, 3]

    if rot_type == 'quat':
        rot = rotation_matrix_to_quaternion(T[:3, :3])
    elif rot_type == 'rpy':
        rot = rotation_matrix_to_rpy(T[:3, :3])
    elif rot_type == '6d':
        rot = pt3d.matrix_to_rotation_6d(
            torch.from_numpy(T[:3, :3]).unsqueeze(0)
        ).numpy().flatten()
    elif rot_type == 'rotvec':  # 新增旋转向量选项
        rot, _ = cv2.Rodrigues(T[:3, :3])  # 将旋转矩阵转换为旋转向量
        rot = rot.flatten()  # 展平为 [rx, ry, rz]
    else:
        raise NotImplementedError("Unsupported rotation type")
    return np.concatenate([pos, rot])


def T_from_6D(T_6D, rot_type):
    pos = T_6D[:3]  # 提取位置部分

    if rot_type == 'quat':
        rot = quaternion_to_rotation_matrix(T_6D[3:])  # 将四元数转换为旋转矩阵
    elif rot_type == 'rpy':
        rot = rpy_to_rotation_matrix(T_6D[3:])  # 将RPY转换为旋转矩阵
    elif rot_type == '6d':
        rot = pt3d.rotation_6d_to_matrix(torch.from_numpy(T_6D[3:].reshape(1, -1))).numpy().reshape(3, 3)
    elif rot_type == 'rotvec':  # 从旋转向量恢复旋转矩阵
        rot, _ = cv2.Rodrigues(T_6D[3:])  # 从旋转向量转换为旋转矩阵
    else:
        raise NotImplementedError("Unsupported rotation type")
    
    # 构建 4x4 变换矩阵
    T_matrix = np.eye(4)
    T_matrix[:3, 3] = pos  # 设置位置
    T_matrix[:3, :3] = rot  # 设置旋转部分
    
    return T_matrix

def one_stage_deploy(args, cfg, tz, bert, task_emb, rob=None, robotiqgrip=None):
    # 设置输入轨迹
    simulate_poses = np.load(f"simulate_teapot_coaster/pose{args.pose}.npy")    # 注意其本身形式就是6D的


    time_idx = 0
    
    # 机械臂相对于基座位置
    T_w_g_init = np.array(
[[-0.020265, -0.999305 ,-0.031289 , 0.555937],
 [-0.994957,  0.017082,  0.098836, -0.090992],
 [-0.098233 , 0.033134, -0.994612 ,-0.013024],
 [ 0. , 0. , 0,  1]]
    )


    now_gripper_pos = np.array(rob.getl())
    pos = now_gripper_pos[:3]
    rot_vector = now_gripper_pos[3:]
    # 将旋转向量转换为旋转矩阵
    rot_matrix, _ = cv2.Rodrigues(rot_vector)
    # 创建 4x4 变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rot_matrix
    transform_matrix[:3, 3] = pos
    np.set_printoptions(precision=6, suppress=True, floatmode='fixed')
    print(transform_matrix)
    T_w_g_init = transform_matrix

    # 相机的手眼标定
#     T_w_cam_init = np.array([
#     [-0.02037987 , 0.53930541 ,-0.84186361 , 1.05595783],
#     [ 0.99751462 ,-0.0458375,  -0.05351179 ,-0.00755727],
#     [-0.06744812 ,-0.84086182 ,-0.53703086 , 0.31069273],
#     [ 0.     ,     0.    ,      0.      ,    1.        ]
#  ])
    
    # 获取参考和物体的初始位姿
    T_cam_o_init = np.load("pose/obj_pose.npy")
    T_cam_ref_init = np.load("pose/ref_pose.npy")
    print(f"T_cam_o_init:{T_cam_o_init}")
    print(f"T_cam_ref_init:{T_cam_ref_init}")


    # global relevant_pose_T      # 利于接收位姿
    # while relevant_pose_T is None:
    # # while True:
    #     print(relevant_pose_T)
    #     time.sleep(0.001)
    
    """
    o_w @ w_g = o_g:得到夹爪相对于物体的初始位姿
    T_w_cam_init 提前手眼标定得到的
    T_cam_o_init FP读到的
    T_w_g_init 机械臂读到的
    """
    T_o_g_init = np.linalg.inv(T_w_cam_init @ T_cam_o_init) @ T_w_g_init    
    
    """
    w_cam @ cam_ref @ ref_o @ o_g = w_g:得到夹爪相对于基座的新位姿
    T_w_cam_init 提前手眼标定得到的
    T_cam_ref_init FP读到的
    relevant_pose_T 实时更新的
    T_o_g_init 刚刚计算出来的
    """
    T_w_g = T_w_cam_init @ T_cam_ref_init @ T_from_6D(simulate_poses[0], '6d') @ T_o_g_init
    # relevant_pose_T = T_ref_objT_w_g_init

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_timesteps = 2000
    num_stages = 1

    rot_type = cfg.rot_type
    hist_len = cfg.hist_len
    
    # 读取语义向量（暂时无用）
    task_emb = torch.from_numpy(task_emb).float().to(device)

    # 初始化模型
    arm_model, gripper_model = load_policy(cfg, device)     # arm_model为生成机械臂运动，gripper_model为生成夹爪运动
    DDIM = DDIMScheduler(**cfg.ddim_cfg)    # 实例化一个DDIM调度器
    DDIM.set_timesteps(cfg.eval_timesteps)  # 指定时间步数，使用多少步从初始噪声还原到目标状态
    DDIM.alphas_cumprod = (DDIM.alphas_cumprod.to(device))      # 将DDIM参数转移到GPU上

    act_trunk = arm_model.act_trunk     # act_trunk为模型的一个参数：用于表示模型一次生成的联系动作步数
    input_dim = arm_model.input_dim     # 获取机械臂使用的向量维度
    all_time_actions = np.zeros([max_timesteps, max_timesteps+act_trunk, input_dim])    # 初始化轨迹的数据保存（置0）

    '''
    初始化位置
    '''
    relative_pose = simulate_poses[0]  # 此处为6d的形式
    # relative_pose = T_to_6D(relevant_pose_T, rot_type="6d")  # 把相对位置转化为6d的坐标形式
    hist_obs = torch.zeros([hist_len, input_dim]).unsqueeze(0).to(device).float()   # 初始化历史观测向量，长度为his_len
    hist_obs[-1] = torch.from_numpy(relative_pose).to(device).float()   # 将目前的相对位姿记录进最新的历史
    # hist_obs[0, -1, :] = torch.from_numpy(relative_pose).to(device).float()

    gripper_state = np.array([0])   
    stage_change = torch.zeros(1).to(device).long() # 初始化阶段变化向量
    state = torch.tensor([0]).to(device).long()     # 初始化状态向量

    is_done = False
    smooth_idx = 0

    generated_trajectory = []

    for time_index, relevant_pose in enumerate(simulate_poses[1: ]):
    # for i in tqdm(range(5)):

        encode_cache = arm_model.forward_enc(hist_obs, task_emb, state)     # 将历史信息、任务语义、当前状态输入进编码器

        # 随机生成一个噪声，作为扩散模型的初始输入
        noicy_action = torch.randn((1, act_trunk, input_dim)).to(device)
        # 扩散模型的逐步去噪过程
        for timestep in DDIM.timesteps: 
            # 构造时间步信息，时间步就理解成训练过程中加噪的“次”
            batched_timestep = timestep.repeat(noicy_action.shape[0]).to(device)

            # 利用解码器，计算噪声残差
            noise_pred = arm_model.forward_dec(noicy_action, encode_cache, batched_timestep)

            # 根据噪声残差、时间步信息，将噪声还原成目标信息
            noicy_action = DDIM.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=noicy_action
            ).prev_sample


        ### 以下就是利用扩散模型的生成，来获取并得到相关轨迹
        arm = noicy_action.detach().cpu().numpy().squeeze() # 提取预测动作到arm，arm实际上为act_trunk个动作
        all_time_actions[smooth_idx][smooth_idx:smooth_idx+act_trunk] = arm     # 记录预测动作
        
        # smooth_action = action_smooth(all_time_actions, smooth_idx)     # 对累计的动作做了一个平滑处理

        # smooth_action = arm[0]      # 生成的第一个动作序列给smooth_action Why???

        # for smooth_action in arm[9:10]:
        for smooth_action in arm[0:1]:
        # for smooth_action in arm:
            # smooth_action = arm  # 仅针对act_trunk=1时

            stage_change = gripper_model(hist_obs, task_emb, state)     # 判断夹爪状态（暂时无用）
            stage_change = stage_change.argmax().item()     # 得到返回值

            pos = smooth_action[:3]
            rot = pt3d.rotation_6d_to_matrix(
                torch.from_numpy(smooth_action[3:]).unsqueeze(0)
            ).numpy()
            # print("heihei")
            # print(pos)
            # print(rot)
            
            T_ref_o = np.eye(4)
            T_ref_o[:3, :3] = rot
            T_ref_o[:3, 3] = pos
            
            """
            o_w @ w_g = o_g:得到夹爪相对于物体的初始位姿
            T_w_cam_init 提前手眼标定得到的
            T_cam_o_init 物体相对于相机的初始位姿
            T_w_g_init 夹爪相对于基座的初始位姿
            """
            # T_o_g_init = np.linalg.inv(T_w_cam_init @ T_cam_o_init) @ T_w_g_init # o is caozuowuti, ref is cankaowuti

            """
            w_cam @ cam_ref @ ref_o @ o_g = w_g:得到夹爪相对于基座的新位姿
            T_w_cam_init 提前手眼标定得到的
            T_cam_ref_init FP读到的
            relevant_pose_T 实时更新的相对位姿
            T_o_g_init 计算出来的夹爪相对于物体的位姿（移动过程中保持不变的）
            """
            T_w_g = T_w_cam_init @ T_cam_ref_init @ T_ref_o @ T_o_g_init
            T_cam_g = T_cam_ref_init @ T_ref_o @ T_o_g_init
            T_cam_o = T_cam_ref_init @ T_ref_o

            msg = Matrix4x4Flat()
            msg.matrix = T_w_g.flatten().tolist()  # 将矩阵数据转化为列表
            lc.publish("action", msg.encode())

            # generated_trajectory.append((pos, rot))  # 3, 3x3
            # generated_trajectory.append((T_w_g[:3, 3], T_w_g[:3, :3]))  # 3, 3x3
            
            
            if DEBUG:
                # 正常流程
                if time_idx == 0:
                    print(T_w_g)
                T_w_g_6D = T_to_6D(T_w_g, rot_type="rotvec")
                # rob.movel(T_w_g_6D, vel = 0.5)

                # 还原回来
                T_w_g = T_from_6D(T_w_g_6D, rot_type="rotvec")
                T_cam_g = np.linalg.inv(T_w_cam_init) @ T_w_g 

                generated_trajectory.append(T_cam_o)

                # 只需要更新移动物体的姿态，但是这种当参考物体发生移动的时候有问题
                relative_pose = relevant_pose  

                # 保存历史记录
                hist_obs = torch.cat([
                    hist_obs[:, 1:],
                    torch.from_numpy(relative_pose).unsqueeze(0).unsqueeze(0).to(device)
                ], dim=1).float()


        smooth_idx += 1
        time_idx += 1
        # print("!!!!!!!!!!!!")
        # print(stage_change)

        if is_done:
            break

    
    data_poses = load_all_npy_files("simulate_teapot_coaster")   # 读到的是ref_o
    data_poses = [T_cam_ref_init @ pose for pose in data_poses]
    

    # data_poses = [[]]
    # for simulate_pose in simulate_poses:    # simulate_poses为物体相对于参考物体ref_o
    #     simulate_T_cam_o = T_cam_ref_init @ T_from_6D(simulate_pose, '6d')
    #     data_poses[0].append(simulate_T_cam_o)
    
    visualize_T_b_o(generated_trajectory, data_poses)
    visualize_generate_pose(generated_trajectory)
    # plot_stepwise_trajectory(generated_trajectory, None)


def load_all_npy_files(folder_path):
    npy_files = []  # 用来存储加载的 .npy 文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):  # 只处理 .npy 文件
            file_path = os.path.join(folder_path, file_name)  # 获取文件的完整路径
            npy_data = np.load(file_path)  # 加载 .npy 文件
            npy_data = [T_from_6D(pose, '6d') for pose in npy_data]
            npy_files.append(npy_data)  # 将加载的数据添加到列表中

    return npy_files

def parse_args():
    parser = argparse.ArgumentParser(description="RLBench Dataset Generator")
    parser.add_argument('--save_path', '-s', type=str, default='./video_save', help='Where to save the demos.')
    parser.add_argument('--ckpt_dir', '-c', default='results/coordiff_real/coordiff_mlp_rms_leaf/teapot_coaster_single_01234',type=str, help='checkpoint dir.')
    parser.add_argument('--tasks', nargs='*', default=['close_jar'], help='The tasks to collect. If empty, all tasks are collected.')
    parser.add_argument('--image_size', nargs=2, type=int, default=[128, 128], help='The size of the images to save.')
    parser.add_argument('--variations', type=int, default=1, help='Number of variations to collect per task. -1 for all.')
    parser.add_argument('--pose', '-p', type=int, default=0)
    return parser.parse_args()


def main():
    if DEBUG:
        rob = urx.Robot("192.168.101.101")
        # rob.set_tcp((0,0,0.19,0, 0,0))   # tool center point
        time.sleep(1) 
        now_gripper_pos = np.array(rob.getl())
        print("Current tool pose is: ",  now_gripper_pos)
        print("Current joint pose is: ",  rob.getj())
        
        if DEBUG == 2:
            joint_angles = np.array([0., -90, -60, -120, 90, 0])
            joint_angles = joint_angles / 180 * 3.14159
            rob.movej(joint_angles)
            return
        robotiqgrip = Robotiq_Two_Finger_Gripper(rob)
        print("close")
        # robotiqgrip.close_gripper()
        # return

    args = parse_args()
    
    tz, bert = init_bert()
    # 初始化 Hydra
    with initialize(version_base="1.3", config_path=args.ckpt_dir):
        # 加载配置文件
        cfg = compose(config_name="config")

    cfg.arm_model_path = os.path.join(args.ckpt_dir, 'arm_model_last.ckpt')
    cfg.gripper_model_path = os.path.join(args.ckpt_dir, 'gripper_model_last.ckpt')
   
    # test(args, cfg, tz, bert)
    # task_emb = np.load("coordiff_real_world/recollection_final/train/put_task_emb.npy")
    # task_emb = np.load("coordiff_real_world/recollection_final/train/put_task_emb.npy")
    # task_emb = np.load("coordiff_real_world/recollection_final/train/put_new/episode_0/0000/task_language_embed.npy")
    task_emb = np.load("coordiff_real_world/recollection_final/train/teapot_coaster_single_01234/episode_0/0000/task_language_embed.npy")

    if DEBUG:
        one_stage_deploy(args, cfg, tz, bert, task_emb, rob, robotiqgrip)
    else:
        one_stage_deploy(args, cfg, tz, bert, task_emb)

    print('Finish')


def get_relevant_pose(poses, index):
    return poses[index]


if __name__ == '__main__':
    main()