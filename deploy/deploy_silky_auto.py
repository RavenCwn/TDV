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
import math
from T import *
lc = lcm.LCM()

DEBUG = 1 # 0：机器人不动；1：机器人动；2: 复位；

relevant_pose_T = None

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


def get_T_b_g(rob):
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
    T_b_g = transform_matrix
    return T_b_g


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

def one_stage_deploy(args, cfg, tz, bert, task_emb, T_cam_o_init, T_cam_ref_init, rob=None, robotiqgrip=None):
    ### 初始化工作
#     T_b_g_init = np.array(
# [[-0.020265, -0.999305 ,-0.031289 , 0.555937],
#  [-0.994957,  0.017082,  0.098836, -0.090992],
#  [-0.098233 , 0.033134, -0.994612 ,-0.013024],
#  [ 0. , 0. , 0,  1]]
#     )
    now_gripper_pos = np.array(rob.getl())
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
    T_b_g_init = transform_matrix


    # 相机的手眼标定
#     T_b_cam_init = np.array([
# [ 0.00824559,  0.55120538, -0.83432886,  1.02815004],
#  [ 0.99795435, -0.05743326, -0.02808102, -0.00693304],
#  [-0.06339664, -0.83239056, -0.55055137,  0.39449762],
#  [ 0.        ,  0.        ,  0.        ,  1.        ],
#  ])
    
    # 如果不是小东西，则更新抓取后的位姿，否则就沿用之前所读取到的物体位姿
    if not args.tiny:
        time.sleep(0.5)  # 等待新的位姿更新
        T_cam_o_init = np.load("pose/obj_pose.npy")

    
    print(f"T_cam_o_init:{T_cam_o_init}")
    print(f"T_cam_ref_init:{T_cam_ref_init}")


    # '''机械臂移动至抓取位置'''
    # if args.name == 'teapot':
    #     T_o_hand = T_teapot_hand
    # if args.name == 'bluemug':
    #     T_o_hand = T_bluemug_hand
    # else:
    #     print("无效物体")

    # T_o_g_init_action = T_o_hand @ T_hand_g_init
    # T_b_g_action = T_b_cam_init @ T_cam_o_init @ T_o_g_init_action
    # rob.movel(T_b_g_action, wait = True, vel = 0.5)
    # print("已移动到待定抓取位置")


#     T_cam_o_init = np.array(
# [[-0.05094842, -0.65924483,  0.75020083, -0.02193002],
#  [ 0.825458  , -0.45062694, -0.33993239,  0.01750604],
#  [ 0.56215897,  0.60194066,  0.56713811,  0.52953911],
#  [ 0.        ,  0.        ,  0.        ,  1.        ]]
#     )

#     T_cam_ref_init = np.array(
# [[ 0.01924219, -0.99670952, -0.07873198,  0.1772099 ],
#  [-0.81035815, -0.06167155,  0.58268027,  0.03968789],
#  [-0.58561874,  0.05258903, -0.80887838,  0.54056682],
#  [ 0.        ,  0.        ,  0.        ,  1.        ]]
#     )

    global relevant_pose_T      # 利于接收位姿
    while relevant_pose_T is None:
    # while True:
        print(relevant_pose_T)
        time.sleep(0.001)
    
    """
    o_w @ w_g = o_g:得到夹爪相对于物体的初始位姿,定值
    T_b_cam_init 提前手眼标定得到的，定值
    T_cam_o_init FP读到的初始,定值
    T_b_g_init 机械臂初始读到的，定值

    结果:T_o_g_init:夹爪相对于物体的初始位姿，定值
    """
    T_o_g_init = np.linalg.inv(T_b_cam_init @ T_cam_o_init) @ T_b_g_init    
    
    """
    w_cam @ cam_ref @ ref_o @ o_g = w_g:得到夹爪相对于基座的新位姿
    T_b_cam_init 提前手眼标定得到的
    T_cam_ref_init FP读到的
    relevant_pose_T 实时更新的
    T_o_g_init 刚刚计算出来的
    """
    # T_b_g = T_b_cam_init @ T_cam_ref_init @ relevant_pose_T @ T_o_g_init

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
    夹爪必须放在这个之后？
    是因为夹的一瞬间会让它位置变吗？
    '''

    '''
    这么计算，相当于依赖项只会有ref物体,跟踪不在夹爪上，精度会高很多吧
    T_ref_o = inv(T_b_ref) @ T_b_g @ inv(T_o_g)
    T_b_ref = T_b_cam @ T_cam_ref_init
    T_b_g
    T_o_g
    T_cam_ref_init
    T_b_cam
    '''
    T_b_ref = T_b_cam_init @ T_cam_ref_init
    T_b_g_now = get_T_b_g(rob=rob)
    T_ref_o_realtime = np.linalg.inv(T_b_ref) @ T_b_g_now @ np.linalg.inv(T_o_g_init)
    relative_pose = T_to_6D(T_ref_o_realtime, rot_type="6d")  # 把相对位置转化为6d的坐标形式
    # time.sleep(1)

    hist_obs = torch.zeros([hist_len, input_dim]).unsqueeze(0).to(device).float()   # 初始化历史观测向量，长度为his_len
    hist_obs[-1] = torch.from_numpy(relative_pose).to(device).float()   # 将目前的相对位姿记录进最新的历史

    # 初始化历史观测向量，长度为 hist_len，大小为 (1, hist_len, input_dim)
    # hist_obs = torch.zeros([1, hist_len, input_dim]).to(device).float()

    # # 只改变最后一帧
    # hist_obs[0, -1, :] = torch.from_numpy(relative_pose).to(device).float()


    stage_change = torch.zeros(1).to(device).long() # 初始化阶段变化向量
    state = torch.tensor([0]).to(device).long()     # 初始化状态向量

    is_done = False
    smooth_idx = 0

    generated_trajectory = []

    
    for i in tqdm(range(max_timesteps)):
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

        for smooth_action in arm[9:10]:
        # for smooth_action in arm[0:1]:
        # for smooth_action in arm:
            # smooth_action = arm  # 仅针对act_trunk=1时

            stage_change = gripper_model(hist_obs, task_emb, state)     # 判断夹爪状态（暂时无用）
            stage_change = stage_change.argmax().item()     # 得到返回值
            print(f'stage_change:{stage_change}')


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
            T_b_cam_init 提前手眼标定得到的
            T_cam_o_init 物体相对于相机的初始位姿
            T_b_g_init 夹爪相对于基座的初始位姿
            """
            # T_o_g_init = np.linalg.inv(T_b_cam_init @ T_cam_o_init) @ T_b_g_init # o is caozuowuti, ref is cankaowuti

            """
            w_cam @ cam_ref @ ref_o @ o_g = w_g:得到夹爪相对于基座的新位姿
            T_b_cam_init 提前手眼标定得到的
            T_cam_ref_init FP读到的
            relevant_pose_T 实时更新的相对位姿
            T_o_g_init 计算出来的夹爪相对于物体的位姿（移动过程中保持不变的）
            """
            T_b_g_action = T_b_cam_init @ T_cam_ref_init @ T_ref_o @ T_o_g_init


            msg = Matrix4x4Flat()
            msg.matrix = T_b_g_action.flatten().tolist()  # 将矩阵数据转化为列表
            lc.publish("action", msg.encode())

            # generated_trajectory.append((pos, rot))  # 3, 3x3
            
            
            if DEBUG:
                print(T_b_g_action)

                T_b_g_6D = T_to_6D(T_b_g_action, rot_type="rotvec")
                # print(T_b_g_6D)1
                generated_trajectory.append((T_b_g_action[:3, 3], T_b_g_action[:3, :3]))  # 3, 3x3
                # if i > hist_len - 2 :
                if i:
            
                    if stage_change:
                        is_done = True                        # robotiqgrip.open_gripper()
                        break

                    
                    # rob.movel(T_b_g_6D, vel=0.5)  # 执行
                    rob.movel(T_b_g_6D, acc=0.3, vel=0.5)  # 执行

                    # print("Current tool pose is: ",  rob.getl())
                    # print("Current joint pose is: ",  rob.getj())
                    # input()

                    # 只需要更新移动物体的姿态，但是这种当参考物体发生移动的时候有问题
                    T_b_ref = T_b_cam_init @ T_cam_ref_init
                    T_b_g_now = get_T_b_g(rob=rob)
                    T_ref_o_realtime = np.linalg.inv(T_b_ref) @ T_b_g_now @ np.linalg.inv(T_o_g_init)
                    relative_pose = T_to_6D(T_ref_o_realtime, rot_type="6d")  # 把相对位置转化为6d的坐标形式

                    ## 夹爪放开条件
                    # distance =  math.sqrt(relative_pose[0]**2 + relative_pose[1]**2 + relative_pose[2]**2)
                    # if distance <= 0.034:
                    #     is_done = True
                    #     robotiqgrip.open_gripper()
                    #     break

                    # 保存历史记录
                    hist_obs = torch.cat([
                        hist_obs[:, 1:],
                        torch.from_numpy(relative_pose).unsqueeze(0).unsqueeze(0).to(device)
                    ], dim=1).float()


                else:
                    # 只需要更新移动物体的姿态，但是这种当参考物体发生移动的时候有问题
                    T_b_ref = T_b_cam_init @ T_cam_ref_init
                    T_b_g_simulate_now = T_b_g_action
                    T_ref_o_simulate_realtime = np.linalg.inv(T_b_ref) @ T_b_g_simulate_now @ np.linalg.inv(T_o_g_init)
                    relative_pose = T_to_6D(T_ref_o_simulate_realtime, rot_type="6d")  # 把相对位置转化为6d的坐标形式

                    # 保存历史记录
                    hist_obs = torch.cat([
                        hist_obs[:, 1:],
                        torch.from_numpy(relative_pose).unsqueeze(0).unsqueeze(0).to(device)
                    ], dim=1).float()


        smooth_idx += 1
        # print("!!!!!!!!!!!!")
        # print(stage_change)

        if is_done:
            break

    # plot_stepwise_trajectory(generated_trajectory, None)


def parse_args():
    parser = argparse.ArgumentParser(description="RLBench Dataset Generator")
    parser.add_argument('--save_path', '-s', type=str, default='./video_save', help='Where to save the demos.')
    parser.add_argument('--ckpt_dir', '-c', default='results/coordiff_real/coordiff_mlp_rms_leaf/put_new6_gen_scale_30_correct',type=str, help='checkpoint dir.')
    parser.add_argument('--tasks', nargs='*', default=['close_jar'], help='The tasks to collect. If empty, all tasks are collected.')
    parser.add_argument('--image_size', nargs=2, type=int, default=[128, 128], help='The size of the images to save.')
    parser.add_argument('--variations', type=int, default=1, help='Number of variations to collect per task. -1 for all.')
    parser.add_argument('--name', '-n', default=None ,type=str)
    parser.add_argument('--tiny', action='store_true', default=False)
    return parser.parse_args()


def main():
    rob = urx.Robot("192.168.101.101")
    # rob.set_tcp((0,0,0.19,0, 0,0))   # tool center point
    time.sleep(1) 
    now_gripper_pos = np.array(rob.getl())
    print("Current tool pose is: ",  now_gripper_pos)
    print("Current joint pose is: ",  rob.getj())
    args = parse_args()

    tz, bert = init_bert()
    # 初始化 Hydra
    with initialize(version_base="1.3", config_path=args.ckpt_dir):
        # 加载配置文件
        cfg = compose(config_name="config")

    # cfg.arm_model_path = os.path.join(args.ckpt_dir, 'arm_model_last.ckpt')
    cfg.arm_model_path = os.path.join(args.ckpt_dir, 'arm_model_best.ckpt')
    # cfg.gripper_model_path = os.path.join(args.ckpt_dir, 'gripper_model_best.ckpt')
    cfg.gripper_model_path = os.path.join(args.ckpt_dir, 'gripper_model_last.ckpt')
    task_emb = np.load("coordiff_real_world/recollection_final/train/put_new/episode_0/0000/task_language_embed.npy")

    '''移动到初始位置'''
    T_cam_ref_init = np.load("pose/ref_pose.npy")   # 移动到初始位置之前读取避免遮挡
    # rob.movel(wait_pose, wait = True, vel = 0.5)  # 先复位
    '''移动到目标位置'''
    T_o_hand = np.load(f"{args.ckpt_dir}/T_o_hand_pose.npy")
    T_cam_o_init = np.load("pose/obj_pose.npy")     # 第一次读物体位置，抓取后还要再读一次

    # T_b_g_action = T_b_cam_init @ T_cam_o_init @ T_o_hand
    T_b_g_action = T_b_cam_init @ T_cam_o_init @ T_o_hand @ T_hand_g_init

    T_b_g_action_6D = T_to_6D(T_b_g_action, rot_type="rotvec")

    # temp:
    T_b_cam_init_6D = T_to_6D(T_b_cam_init, rot_type="rotvec")
    T_cam_o_init_6D = T_to_6D(T_cam_o_init, rot_type="rotvec")
    T_b_o_init_6D = T_to_6D(T_b_cam_init @ T_cam_o_init, rot_type="rotvec")
    T_o_hand_6D = T_to_6D(T_o_hand, rot_type="rotvec")
    T_hand_g_init_6D = T_to_6D(T_hand_g_init, rot_type="rotvec")
    print("----------------------")
    print("T_b_cam_init_6D", T_b_cam_init_6D)
    print("T_cam_o_init_6D", T_cam_o_init_6D)
    print("T_b_o_init_6D", T_b_o_init_6D)
    print("T_o_hand_6D", T_o_hand_6D)
    print("T_hand_g_init_6D", T_hand_g_init_6D)
    print("----------------------")
    # print(T_b_g_action_6D)
    # temp

    T_b_g_action_up5 = T_b_g_action_6D.copy()
    T_b_g_action_up5[2] += 0.1
    rob.movel(T_b_g_action_up5, wait = True, acc=0.4, vel=0.5)
    print(T_b_g_action_up5)
    print(T_b_g_action_6D)

    rob.movel(T_b_g_action_6D, wait = True, acc=0.4, vel=0.5)

    print("已移动到待定抓取位置")

    # 闭合夹爪
    robotiqgrip = Robotiq_Two_Finger_Gripper(rob)
    robotiqgrip.close_gripper()
    print("close")

    # test(args, cfg, tz, bert)
    # task_emb = np.load("coordiff_real_world/recollection_final/train/put_task_emb.npy")
    # task_emb = np.load("coordiff_real_world/recollection_final/train/put_task_emb.npy")
    task_emb = np.load("coordiff_real_world/recollection_final/train/put_new/episode_0/0000/task_language_embed.npy")
    # return
    if DEBUG:
        one_stage_deploy(args, cfg, tz, bert, task_emb, T_cam_o_init, T_cam_ref_init, rob, robotiqgrip)
    else:
        one_stage_deploy(args, cfg, tz, bert, task_emb)

    time.sleep(1)
    rob.movel(T_b_g_action_6D, wait = True, acc=0.3, vel=0.5)
    robotiqgrip.open_gripper()
    rob.movel(T_b_g_action_up5, wait = True, acc=0.3, vel=0.5)
    rob.movel(wait_pose, wait = True, acc=0.3, vel=0.5)
    print('Finish')


if __name__ == '__main__':
    main()
