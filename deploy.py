import numpy as np
import argparse
import os
from PIL import Image

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.backend.utils import task_file_to_task_class

import rlbench.backend.task as task
import hydra
import torch
from easydict import EasyDict
from transformers import AutoTokenizer, AutoModel
from pyrep.objects import Object
from coordiff.utils.transform import *
from coordiff.models import *
from hydra import initialize, compose
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm
import time
import shutil
from eval_utils import *

def test(args, cfg, tz, bert):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_timesteps = 200
    num_stages = 1

    rot_type = cfg.rot_type
    hist_len = cfg.hist_len
    
    relevant_traj = np.load("coordiff_real_world/recollection_data/train/pour_water/episode_4/0000/relevant_traj.npy")
    task_emb = np.load("coordiff_real_world/recollection_data/train/pour_water/episode_4/0000/task_language_embed.npy")
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
    relative_pose = relevant_traj[ref_idx]
    ref_idx += 1

    hist_obs = torch.zeros([hist_len, input_dim]).unsqueeze(0).to(device).float()
    hist_obs[-1] = torch.from_numpy(relative_pose).to(device).float()
    gripper_state = np.array([0])
    stage_change = torch.zeros(1).to(device).long()
    state = torch.tensor([0]).to(device).long()

    done = False
    smooth_idx = 0

    generated_trajectory = []
    gt_trajectory = []
    for ref_T in relevant_traj:
        pos = ref_T[:3]
        rot = pt3d.rotation_6d_to_matrix(
            torch.from_numpy(ref_T[3:]).unsqueeze(0)
        ).numpy()
        gt_trajectory.append((pos, rot))
        print(pos)

    for i in tqdm(range(max_timesteps)):

        encode_cache = arm_model.forward_enc(hist_obs, task_emb, state)

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
        stage_change = gripper_model(hist_obs, task_emb, state)
        stage_change = stage_change.argmax().item()

        pos = smooth_action[:3]
        rot = pt3d.rotation_6d_to_matrix(
            torch.from_numpy(smooth_action[3:]).unsqueeze(0)
        ).numpy()

        generated_trajectory.append((pos, rot))

        smooth_idx += 1


        # if stage_change and state < num_stages:
        #     state += 1
        #     print(f"stage_change: {state}")
        #     gripper_state = 1 - gripper_state
            
        #     if state == num_stages:
        #         break

        # 只需要更新移动物体的姿态，但是这种当参考物体发生移动的时候有问题
        relative_pose = relevant_traj[ref_idx]
        ref_idx += 1
        
        hist_obs = torch.cat([
            hist_obs[:, 1:],
            torch.from_numpy(relative_pose).unsqueeze(0).unsqueeze(0).to(device)
        ], dim=1).float()


    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    # 提取轨迹点
    gen_positions = np.array([pos for pos, _ in generated_trajectory])
    gt_positions = np.array([pos for pos, _ in gt_trajectory])

    # 创建 3D 图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 画轨迹
    ax.plot(gen_positions[:, 0], gen_positions[:, 1], gen_positions[:, 2], markersize=1, marker='o', linestyle='-', color='r', label="Generated Trajectory")
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],  markersize=1, marker='o', linestyle='-', color='b', label="Ground Truth Trajectory")

    # 画坐标轴方向
    axis_length = 0.001  # 增大坐标轴长度，便于观察旋转

    def draw_axes(ax, pos, rot, label_prefix):
        """ 在给定位置绘制局部坐标系 """
        x_axis, y_axis, z_axis = rot[:, 0], rot[:, 1], rot[:, 2]
        
        # X轴 (红色)
        ax.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2], length=axis_length, color='r', linewidth=2)
        ax.text(pos[0] + x_axis[0] * axis_length, pos[1] + x_axis[1] * axis_length, pos[2] + x_axis[2] * axis_length, f'{label_prefix}X', color='r', fontsize=10)

        # Y轴 (绿色)
        ax.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2], length=axis_length, color='g', linewidth=2)
        ax.text(pos[0] + y_axis[0] * axis_length, pos[1] + y_axis[1] * axis_length, pos[2] + y_axis[2] * axis_length, f'{label_prefix}Y', color='g', fontsize=10)

        # Z轴 (蓝色)
        ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2], length=axis_length, color='b', linewidth=2)
        ax.text(pos[0] + z_axis[0] * axis_length, pos[1] + z_axis[1] * axis_length, pos[2] + z_axis[2] * axis_length, f'{label_prefix}Z', color='b', fontsize=10)

    # # 遍历轨迹绘制坐标系
    # for pos, rot in generated_trajectory:
    #     draw_axes(ax, pos, rot[0], "Gen_")

    # for pos, rot in gt_trajectory:
    #     draw_axes(ax, pos, rot[0], "GT_")

    # 轴标签
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Comparison of Generated and Ground Truth Trajectories")

    ax.legend()
    plt.show()

def deploy(args, cfg, tz, bert):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_timesteps = 200
    num_stages = 1

    rot_type = cfg.rot_type
    hist_len = cfg.hist_len
    
    # 初始化抓取相对位置 T_o_g, 从 task-oriented 获得
    T_o_g = None
    task_description = None
           
    # 初始化模型
    arm_model, gripper_model = load_policy(cfg, device)
    DDIM = DDIMScheduler(**cfg.ddim_cfg)
    DDIM.set_timesteps(cfg.eval_timesteps)
    DDIM.alphas_cumprod = (DDIM.alphas_cumprod.to(device))


    act_trunk = arm_model.act_trunk
    input_dim = arm_model.input_dim
    all_time_actions = np.zeros([max_timesteps, max_timesteps+act_trunk, input_dim])


    # 初始化观测
    # TODO 获得参考物体的pose 和 move obj pose
    ref_pose = None  # 执行的时候通过FP获取一次参考物体姿态
    move_pose = None  # 执行的时候通过FP获取一次移动物体姿态
    relative_pose = compute_relative_pose_input(move_pose, ref_pose, rot_type) 


    # hist_obs = [torch.from_numpy(relative_pose) for _ in range(hist_len)]
    # hist_obs = torch.stack(hist_obs, dim=0).unsqueeze(0).to(device).float()
    hist_obs = torch.zeros([hist_len, input_dim]).unsqueeze(0).to(device).float()
    hist_obs[-1] = torch.from_numpy(relative_pose).to(device).float()
    task_emb = get_task_embs(cfg, task_description, tz, bert).to(device).float()
    gripper_state = np.array([0])
    stage_change = torch.zeros(1).to(device).long()



    done = False
    smooth_idx = -1
    for i in tqdm(range(max_timesteps)):

        encode_cache = arm_model.forward_enc(hist_obs, task_emb, state)

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
        stage_change = gripper_model(hist_obs, task_emb, state)
        stage_change = stage_change.argmax().item()

        gripper_pose = get_abs_action(smooth_action, ref_pose, rot_type, T_o_g)

        smooth_idx += 1

        action = np.concatenate([gripper_pose, gripper_state])

        # send action

        if stage_change and state < num_stages:
            state += 1
            print(f"stage_change: {state}")
            gripper_state = 1 - gripper_state
            
            if state == num_stages:
                break
            
            move_obj_name, ref_obj_name = task_stage[state.item()]
            print("move_obj_name: ", move_obj_name)
            print("ref_obj_name: ", ref_obj_name)
            
            # 重新更新下一阶段的抓取姿势、参考物体、目标物体
            # 初始化抓取相对位置 T_o_g, 从 task-oriented 获得
            T_o_g = None
            move_pose = get_abs_pose(move_obj_name, obs, task)
            ref_pose = get_abs_pose(ref_obj_name, obs, task)
            relative_pose = compute_relative_pose_input(move_pose, ref_pose, rot_type)

            hist_obs = [torch.from_numpy(relative_pose) for _ in range(hist_len)]
            hist_obs = torch.stack(hist_obs, dim=0).unsqueeze(0).to(device).float()

        # 只需要更新移动物体的姿态，但是这种当参考物体发生移动的时候有问题
        move_pose = get_pose_for_task(obs, task_name, task_description, move_obj_name, task)
        relative_pose = compute_relative_pose_input(move_pose, ref_pose, rot_type).reshape(1, -1)

        hist_obs = torch.cat([
            hist_obs[:, 1:],
            torch.from_numpy(relative_pose).unsqueeze(0).to(device)
        ], dim=1).float()

def parse_args():
    parser = argparse.ArgumentParser(description="RLBench Dataset Generator")
    parser.add_argument('--save_path', '-s', type=str, default='./video_save', help='Where to save the demos.')
    parser.add_argument('--ckpt_dir', '-c', type=str, help='checkpoint dir.')
    parser.add_argument('--tasks', nargs='*', default=['close_jar'], help='The tasks to collect. If empty, all tasks are collected.')
    parser.add_argument('--image_size', nargs=2, type=int, default=[128, 128], help='The size of the images to save.')
    parser.add_argument('--variations', type=int, default=1, help='Number of variations to collect per task. -1 for all.')
    return parser.parse_args()


def main():
    args = parse_args()
    
    tz, bert = init_bert()

    # 初始化 Hydra
    with initialize(version_base="1.3", config_path=args.ckpt_dir):
        # 加载配置文件
        cfg = compose(config_name="config")

    cfg.arm_model_path = os.path.join(args.ckpt_dir, 'arm_model_best.ckpt')
    cfg.gripper_model_path = os.path.join(args.ckpt_dir, 'gripper_model_best.ckpt')
   
    test(args, cfg, tz, bert)

    print('Finish')


if __name__ == '__main__':
    main()