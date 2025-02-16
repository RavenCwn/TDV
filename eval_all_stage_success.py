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
from coordiff.utils.transfrom import *
from coordiff.models import *
from hydra import initialize, compose
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm

def save(img, path, idx): 
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, f'{idx:04}.png')
    img.save(path) 
    return idx+1

def action():
    # arm = np.random.normal(0.0, 0.1, size=(7,))
    arm = np.array([1.915595978498458862e-01, -8.240001648664474487e-02, 
                8.656496405601501465e-01, -8.218078613281250000e-01, 
                -5.697641372680664062e-01, 6.558554014191031456e-04, 
                5.257503944449126720e-04]) # pose:[x, y, z, qx, qy, qz, qw]
    gripper = [1.0]  # open
    return np.concatenate([arm, gripper], axis=-1)

def load_policy(cfg, device):
    model_cls = eval(cfg.arm_model_name)
    arm_model = model_cls(**cfg.arm_model_cfg, device=device).to(device)
    arm_model.load_state_dict(torch.load(cfg.arm_model_path, map_location=device))

    model_cls = eval(cfg.gripper_model_name)
    gripper_model = model_cls(**cfg.gripper_model_cfg, device=device).to(device)
    gripper_model.load_state_dict(torch.load(cfg.gripper_model_path, map_location=device))

    print("success load model!")

    return arm_model, gripper_model

def action_smooth(all_time_actions, t):
    actions_for_curr_step = all_time_actions[:, t]
    actions_populated = np.all(actions_for_curr_step != 0, axis=1)
    actions_for_curr_step = actions_for_curr_step[actions_populated]
    k = 0.01
    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
    exp_weights = exp_weights / np.sum(exp_weights)
    exp_weights = exp_weights[:, np.newaxis]
    action = np.sum(actions_for_curr_step * exp_weights, axis=0)

    return action

def get_task_embs(cfg, description, tz, model):
    """
    Bert embeddings for task embeddings. Borrow from https://github.com/Lifelong-Robot-Learning/LIBERO/blob/f78abd68ee283de9f9be3c8f7e2a9ad60246e95c/libero/lifelong/utils.py#L152.
    """
    cfg = EasyDict({
        "task_embedding_format": "bert",
        "task_embedding_one_hot_offset": 1,
        "data": {"max_word_len": 25},
        "policy": {"language_encoder": {"network_kwargs": {"input_size": 768}}}
    })  # hardcode the config to get task embeddings according to original Libero code

    if cfg.task_embedding_format == "bert":
        tokens = tz(
            text=description,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
    else:
        raise ValueError("Unsupported task embedding format")
    cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]
    return task_embs

def init_bert():
    tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir="./bert")
    model = AutoModel.from_pretrained(
        "bert-base-cased", cache_dir="./bert")
    return tz, model




def start(task, args, cfg, tz, bert, num_episodes):
    idx = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
        obs_config=ObservationConfig(),
        headless=True)
    rlbench_env.launch()

    task_env = rlbench_env.get_task(task)
    task_env.set_variation(args.variations)

    success_rate = 0

    arm_model, gripper_model = load_policy(cfg, device)
    DDIM = DDIMScheduler(**cfg.ddim_cfg)
    DDIM.set_timesteps(cfg.eval_timesteps)
    DDIM.alphas_cumprod = (
        DDIM.alphas_cumprod.to(device)
    )

    with open(f"task_stage/{task_env._task.get_name()}.txt", "r") as f:
        task_stage = f.readlines()
        task_stage = [x.strip().split(' ') for x in task_stage]
    print("task_stage: ", task_stage)

    for epid in range(num_episodes):
        task_description, obs = task_env.reset()
        

        task = task_env._task
        scene = task_env._scene
        robot = task_env._robot
        random_seed = np.random.get_state()
        
        hist_len = cfg.hist_len
        rot_type = cfg.rot_type



        # 0 时刻的物体
        state = torch.tensor([0]).to(device).long()
        move_obj_name, ref_obj_name = task_stage[state.item()]
        move_pose = get_abs_pose(move_obj_name, obs, task)
        ref_pose = get_abs_pose(ref_obj_name, obs, task)
        relative_pose = compute_relative_pose_input(move_pose, ref_pose, rot_type)

        # 历史观测用开始填充
        hist_obs = [torch.from_numpy(relative_pose) for _ in range(hist_len)]
        hist_obs = torch.stack(hist_obs, dim=0).unsqueeze(0).to(device).float()
        task_emb = get_task_embs(cfg, task_description[0], tz, bert).to(device).float()
        gripper_state = np.array([1])


        done = False
        T_o_g = None
        smooth_idx = -1
        max_timesteps = 220

        
        act_trunk = arm_model.act_trunk
        input_dim = arm_model.input_dim
        all_time_actions = np.zeros([max_timesteps, max_timesteps+act_trunk, input_dim])


        # try:
        for i in range(max_timesteps):
            print(i)
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


            if smooth_idx == -1:
                # gripper_pose = get_abs_action(arm[5], ref_pose, rot_type, T_o_g)
                gripper_pose = obs.gripper_pose
                stage_change = torch.zeros(1).to(device).long()
                print("init")
            else:
                arm = noicy_action.detach().cpu().numpy().squeeze()
                all_time_actions[smooth_idx][smooth_idx:smooth_idx+act_trunk] = arm
                smooth_action = action_smooth(all_time_actions, smooth_idx)
                stage_change = gripper_model(hist_obs, task_emb, state)
                stage_change = stage_change.argmax().item()

                # TODO 获得参考物体的pose
                # gripper_pose = get_abs_action(arm[0], ref_pose, rot_type, T_o_g)
                gripper_pose = get_abs_action(smooth_action, ref_pose, rot_type, T_o_g)
            smooth_idx += 1

            action = np.concatenate([gripper_pose, gripper_state])
            obs, reward, done = task_env.step(action)
            if done:
                break
                
            if stage_change and state.item() < 1:
                state += 1
                print(f"stage_change: {state}")
                gripper_state = 1 - gripper_state
                gripper_pose = obs.gripper_pose
                action = np.concatenate([gripper_pose, gripper_state])
                obs, reward, done = task_env.step(action)
                obs, reward, done = task_env.step(action)
                obs, reward, done = task_env.step(action)
                if done:
                    break

                if move_obj_name == 'gripper_pose':
                    move_pose = get_abs_pose(move_obj_name, obs, task)
                    ref_pose = get_abs_pose(ref_obj_name, obs, task)
                    print("before move_obj_name: ", move_obj_name)
                    print("before ref_obj_name: ", ref_obj_name)
                    T_o_g = compute_relative_pose_T(move_pose, ref_pose,)
                    print("T_o_g: ", T_o_g)

                move_obj_name, ref_obj_name = task_stage[state.item()]
                print("move_obj_name: ", move_obj_name)
                print("ref_obj_name: ", ref_obj_name)
                move_pose = get_abs_pose(move_obj_name, obs, task)
                ref_pose = get_abs_pose(ref_obj_name, obs, task)
                relative_pose = compute_relative_pose_input(move_pose, ref_pose, rot_type)

                # hist_obs = [torch.from_numpy(relative_pose) for _ in range(hist_len)]
                # hist_obs = torch.stack(hist_obs, dim=0).unsqueeze(0).to(device).float()

                hist_obs = torch.zeros([hist_len, input_dim]).unsqueeze(0).to(device).float()
                hist_obs[-1] = torch.from_numpy(relative_pose).to(device).float()

                all_time_actions = np.zeros([max_timesteps, max_timesteps+act_trunk, input_dim])
                smooth_idx = -1
            elif stage_change and state.item() == 1:
                gripper_state = 1 - gripper_state
                gripper_pose = obs.gripper_pose
                state += 1
            else:
                # hist_update
                # 获得运动物体和参考物体的pose
                move_pose = get_abs_pose(move_obj_name, obs, task)
                ref_pose = get_abs_pose(ref_obj_name, obs, task)
                relative_pose = compute_relative_pose_input(move_pose, ref_pose, rot_type).reshape(1, -1)

                hist_obs = torch.cat([
                    hist_obs[:, 1:],
                    torch.from_numpy(relative_pose).unsqueeze(0).to(device)
                ], dim=1).float()


            front_rgb = Image.fromarray(obs.front_rgb)
            path = os.path.join(args.save_path, task.get_name(), f"{epid:04d}", 'front_rgb')
            idx = save(front_rgb, path, idx)
            
            

            if done:
                print("Episode success!")
        # except Exception as e:
        #     print(e)

        success_rate += done

    rlbench_env.shutdown()

    return success_rate / num_episodes


def parse_args():
    parser = argparse.ArgumentParser(description="RLBench Dataset Generator")
    parser.add_argument('--save_path', '-s', type=str, default='./save', help='Where to save the demos.')
    parser.add_argument('--ckpt_dir', '-c', type=str, help='checkpoint dir.')
    parser.add_argument('--tasks', nargs='*', default=['close_jar'], help='The tasks to collect. If empty, all tasks are collected.')
    parser.add_argument('--image_size', nargs=2, type=int, default=[128, 128], help='The size of the images to save.')
    parser.add_argument('--variations', type=int, default=1, help='Number of variations to collect per task. -1 for all.')
    parser.add_argument('--episodes', type=int, default=10, )
    return parser.parse_args()


def main():
    args = parse_args()
    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    if len(args.tasks) > 0:
        for t in args.tasks:
            if t not in task_files:
                raise ValueError('Task %s not recognised!.' % t)
        task_files = args.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    tz, bert = init_bert()

    # 初始化 Hydra
    with initialize(version_base="1.3", config_path=args.ckpt_dir):
        # 加载配置文件
        cfg = compose(config_name="config")

    cfg.arm_model_path = os.path.join(args.ckpt_dir, 'arm_model_1000.ckpt')
    cfg.gripper_model_path = os.path.join(args.ckpt_dir, 'gripper_model_1000.ckpt')

    for t in tasks:
        success_rate = start(t, args, cfg, tz, bert, args.episodes)
        print("Task: ", t, "Success rate: ", success_rate)

    print('Finish')


if __name__ == '__main__':
    main()