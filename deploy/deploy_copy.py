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
lc = lcm.LCM()

DEBUG = 2 # 0：机器人不动；1：机器人动；2: 复位；

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

def one_stage_deploy(args, cfg, tz, bert, task_emb, rob=None, robotiqgrip=None):
    T_w_g_init = np.array(
[[-0.020265, -0.999305 ,-0.031289 , 0.555937],
 [-0.994957,  0.017082,  0.098836, -0.090992],
 [-0.098233 , 0.033134, -0.994612 ,-0.013024],
 [ 0. , 0. , 0,  1]]
    )
    now_gripper_pos = np.array(rob.getl())
    # print("Current tool pose is: ",  now_gripper_pos)
    # 分离位置和旋转向量
    import cv2
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

    T_w_cam_init = np.array([
         [-0.02037987 , 0.53930541 ,-0.84186361 , 1.05595783],
 [ 0.99751462 ,-0.0458375,  -0.05351179 ,-0.00755727],
 [-0.06744812 ,-0.84086182 ,-0.53703086 , 0.31069273],
 [ 0.     ,     0.    ,      0.      ,    1.        ]
 ])
       
    T_cam_o_init = np.load("pose/obj_pose.npy")
    T_cam_ref_init = np.load("pose/ref_pose.npy")

    print(f"T_cam_o_init:{T_cam_o_init}")
    print(f"T_cam_ref_init:{T_cam_ref_init}")

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

    global relevant_pose_T

    while relevant_pose_T is None:
    # while True:
        print(relevant_pose_T)
        time.sleep(0.001)
    T_o_g_init = np.linalg.inv(T_w_cam_init @ T_cam_o_init) @ T_w_g_init
    print(T_w_g_init)
    T_w_g = T_w_cam_init @ T_cam_ref_init @ relevant_pose_T @ T_o_g_init
    # relevant_pose_T = T_ref_objT_w_g_init
    print(f"T_w_g_init: {T_w_g_init}")
    print(f"T_w_g: {T_w_g}")

    # rob.movep(T_w_g)
    # return
    # T_w_g = T_w_cam_init @ T_cam_ref_init @ relevant_pose_T

    # if DEBUG:
    #     print(T_w_g)
    #     T_w_g_6D = T_to_6D(T_w_g, rot_type="rotvec")
    #     print("!!!!")
    #     print(T_w_g_6D)
    #     rob.movep(T_w_g_6D)
    #     print("Current tool pose is: ",  rob.getl())
    #     print("Current joint pose is: ",  rob.getj())
    #     # input()
    #     # robotiqgrip.close_gripper()
    #     input()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_timesteps = 2000
    num_stages = 1

    rot_type = cfg.rot_type
    hist_len = cfg.hist_len
    
    # task_emb = np.load("coordiff_real_world/recollection_data_smooth5/train/pour_water/episode_0/0000/task_language_embed.npy")
    task_emb = torch.from_numpy(task_emb).float().to(device)

    # 初始化模型
    arm_model, gripper_model = load_policy(cfg, device)
    DDIM = DDIMScheduler(**cfg.ddim_cfg)
    DDIM.set_timesteps(cfg.eval_timesteps)
    DDIM.alphas_cumprod = (DDIM.alphas_cumprod.to(device))

    act_trunk = arm_model.act_trunk
    input_dim = arm_model.input_dim
    all_time_actions = np.zeros([max_timesteps, max_timesteps+act_trunk, input_dim])

    ref_idx = 0
    ref_idx += 1

    relative_pose = T_to_6D(relevant_pose_T, rot_type="6d")  # transfer to 6d
    hist_obs = torch.zeros([hist_len, input_dim]).unsqueeze(0).to(device).float()
    hist_obs[-1] = torch.from_numpy(relative_pose).to(device).float()
    gripper_state = np.array([0])
    stage_change = torch.zeros(1).to(device).long()
    state = torch.tensor([0]).to(device).long()

    done = False
    smooth_idx = 0

    generated_trajectory = []

    for i in tqdm(range(max_timesteps)):
    # for i in tqdm(range(5)):

        encode_cache = arm_model.forward_enc(hist_obs, task_emb, state)

        # 随机生成一个噪声，作为扩散模型的初始输入
        noicy_action = torch.randn((1, act_trunk, input_dim)).to(device)
        for timestep in DDIM.timesteps:
            # predict noise given timestep
            batched_timestep = timestep.repeat(noicy_action.shape[0]).to(device)

            noise_pred = arm_model.forward_dec(noicy_action, encode_cache, batched_timestep)

            # take diffusion step
            noicy_action = DDIM.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=noicy_action
            ).prev_sample


        arm = noicy_action.detach().cpu().numpy().squeeze()
        all_time_actions[smooth_idx][smooth_idx:smooth_idx+act_trunk] = arm
        smooth_action = action_smooth(all_time_actions, smooth_idx)
        smooth_action = arm[0]
        stage_change = gripper_model(hist_obs, task_emb, state)
        stage_change = stage_change.argmax().item()

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
        
        T_o_g_init = np.linalg.inv(T_w_cam_init @ T_cam_o_init) @ T_w_g_init # o is caozuowuti, ref is cankaowuti

        T_w_g = T_w_cam_init @ T_cam_ref_init @ T_ref_o @ T_o_g_init
        msg = Matrix4x4Flat()
        msg.matrix = T_w_g.flatten().tolist()  # 将矩阵数据转化为列表
        lc.publish("action", msg.encode())

        # print(T_w_g)

        # generated_trajectory.append((pos, rot))  # 3, 3x3
        generated_trajectory.append((T_w_g[:3, 3], T_w_g[:3, :3]))  # 3, 3x3
        
        if i%1 == 0 and DEBUG:
            print(T_w_g)
            T_w_g_6D = T_to_6D(T_w_g, rot_type="rotvec")
            print(T_w_g_6D)
            rob.movel(T_w_g_6D)
            print("Current tool pose is: ",  rob.getl())
            # print("Current joint pose is: ",  rob.getj())
            # input()
        smooth_idx += 1
        # if stage_change and state < num_stages:
        #     state += 1
        #     print(f"stage_change: {state}")
        #     gripper_state = 1 - gripper_state
            
        #     if state == num_stages:
        #         break
        print("!!!!!!!!!!!!")
        print(stage_change)
        # if stage_change:
        #     robotiqgrip.open_gripper()


        # 只需要更新移动物体的姿态，但是这种当参考物体发生移动的时候有问题
        relative_pose = T_to_6D(relevant_pose_T, rot_type="6d")  
        
        ## 跳出条件
        distance =  math.sqrt(relative_pose[0]**2 + relative_pose[1]**2 + relative_pose[2]**2)

        if distance <= 0.034:
            robotiqgrip.open_gripper()
            break

        # transfer to 6d
        ref_idx += 1
        # print("------")
        # print(relative_pose)
        # print(relevant_pose_T)
        # print("------")

        # time.sleep(1)

        hist_obs = torch.cat([
            hist_obs[:, 1:],
            torch.from_numpy(relative_pose).unsqueeze(0).unsqueeze(0).to(device)
        ], dim=1).float()

    plot_stepwise_trajectory(generated_trajectory, None)

def parse_args():
    parser = argparse.ArgumentParser(description="RLBench Dataset Generator")
    parser.add_argument('--save_path', '-s', type=str, default='./video_save', help='Where to save the demos.')
    parser.add_argument('--ckpt_dir', '-c', default='results/coordiff_real/coordiff_mlp_rms_leaf/gen',type=str, help='checkpoint dir.')
    parser.add_argument('--tasks', nargs='*', default=['close_jar'], help='The tasks to collect. If empty, all tasks are collected.')
    parser.add_argument('--image_size', nargs=2, type=int, default=[128, 128], help='The size of the images to save.')
    parser.add_argument('--variations', type=int, default=1, help='Number of variations to collect per task. -1 for all.')
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
            rob.movej(joint_angles, acc=0.2, vel=0.3)
            return
        robotiqgrip = Robotiq_Two_Finger_Gripper(rob)
        print("close")
        robotiqgrip.close_gripper()
        return

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
    task_emb = np.load("coordiff_real_world/recollection_final/train/put_new_gen/episode_0/0000/task_language_embed.npy")

    if DEBUG:
        one_stage_deploy(args, cfg, tz, bert, task_emb, rob, robotiqgrip)
    else:
        one_stage_deploy(args, cfg, tz, bert, task_emb)

    print('Finish')


if __name__ == '__main__':
    main()