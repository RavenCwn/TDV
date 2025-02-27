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

colors = [
    ('red', (1.0, 0.0, 0.0)),
    ('maroon', (0.5, 0.0, 0.0)),
    ('lime', (0.0, 1.0, 0.0)),
    ('green', (0.0, 0.5, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('navy', (0.0, 0.0, 0.5)),
    ('yellow', (1.0, 1.0, 0.0)),
    ('cyan', (0.0, 1.0, 1.0)),
    ('magenta', (1.0, 0.0, 1.0)),
    ('silver', (0.75, 0.75, 0.75)),
    ('gray', (0.5, 0.5, 0.5)),
    ('orange', (1.0, 0.5, 0.0)),
    ('olive', (0.5, 0.5, 0.0)),
    ('purple', (0.5, 0.0, 0.5)),
    ('teal', (0, 0.5, 0.5)),
    ('azure', (0.0, 0.5, 1.0)),
    ('violet', (0.5, 0.0, 1.0)),
    ('rose', (1.0, 0.0, 0.5)),
    ('black', (0.0, 0.0, 0.0)),
    ('white', (1.0, 1.0, 1.0)),
]


def get_color_name(rgb):
    """
    根据输入的RGB浮点值找到最接近的颜色名称。
    
    参数:
        rgb (tuple): 包含三个浮点数的元组，表示RGB颜色值，范围为[0.0, 1.0]。
    
    返回:
        str: 最接近的颜色名称。
    """
    min_distance = float('inf')  # 初始化最小距离为无穷大
    closest_color = None  # 初始化最接近的颜色名称为空

    for color_name, color_rgb in colors:
        # 计算输入RGB与当前颜色的欧几里得距离
        distance = np.sqrt((rgb[0] - color_rgb[0]) ** 2 +
                            (rgb[1] - color_rgb[1]) ** 2 +
                            (rgb[2] - color_rgb[2]) ** 2)
        
        # 如果当前距离小于最小距离，更新最小距离和最接近的颜色名称
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color


def start(rlbench_env, task_class, args, cfg, tz, bert, task_str=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    success = 0
    test_iters = args.iters
    iters = 0
    test_type = "random"  # dataset
    success = 0
    task_env = rlbench_env.get_task(task_class)
    task_name = task_env._task.get_name()
    max_timesteps = 200
    
    print("[Task]: ", task_name)
    # 添加日志文件
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{task_name}_results.log")

    variation_count = task_env._task.variation_count()
    while iters < test_iters:
        try:
            iters_log = f"[iters]: {iters}, success rate: {success}/{iters}"
            print(iters_log)
            with open(log_file, 'a') as f:
                f.write(iters_log + '\n')

            if test_type == "dataset":
                variation = iters % variation_count
                demo = task_env.get_demos(
                    amount=1,
                    from_episode_number=iters,
                    random_selection=False
                )
                task_description, obs = task_env.reset_to_demo(demo)
            elif test_type == "random":
                variation = np.random.randint(0, variation_count)
                task_env.set_variation(variation)
                task_description, obs = task_env.reset()
            else:
                raise NotImplementedError

            print(f"task_description: {task_description}")
            task_description = task_description[0]
                
            waypoints = task_env._task.get_waypoints()
            for i, point in enumerate(waypoints[:2]):
                run_waypoint(point, task_env)
                obs = task_env.get_observation()

            task = task_env._task
            scene = task_env._scene
            robot = task_env._robot
            random_seed = np.random.get_state()
            hist_len = cfg.hist_len
            rot_type = cfg.rot_type

            # 初始化任务阶段
            with open(f"examples/task_stages/{task_name}.txt", "r") as f:
                task_stage = f.readlines()
                task_stage = [x.strip().split(' ') for x in task_stage]
            print("task_stage: ", task_stage)

            state = torch.tensor([0]).to(device).long()
            task_related_obj_list = task_stage[state.item()]
            if task_related_obj_list[2] != "None":
                target_color = None
                for color in colors:
                    if color[0] in task_description:
                        target_color = color[0]
                        break
                assert target_color is not None, f"Select color error. {task_description}, {color[0]}"

                idx = 0 if task_related_obj_list[2] == 'move_obj' else 1
                target_name = task_related_obj_list[idx]
                for obj, _ in task._initial_objs_in_scene:
                    obj_name = obj.get_name()
                    if hasattr(obj, "get_color"):
                        color = obj.get_color()
                        obj_color = get_color_name(color)
                        if (target_name in obj_name) and (obj_color == target_color):
                            task_related_obj_list[idx] = obj_name


            if "light_bulb_in" == task_name:
                target_bulb_holder = get_abs_pose(task_related_obj_list[idx], obs, task)
                bulb0 = get_abs_pose("bulb0", obs, task)
                bulb1 = get_abs_pose("bulb1", obs, task)
                if np.linalg.norm(bulb0[:3] - target_bulb_holder[:3]) < np.linalg.norm(bulb1[:3] - target_bulb_holder[:3]):
                    task_related_obj_list[idx] = "bulb0"
                else:
                    task_related_obj_list[idx] = "bulb1"

            elif "put_money_in_safe" == task_name:
                if "top" in task_description:
                    task_related_obj_list[1] = "dummy_shelf2"
                elif "middle" in task_description:
                    task_related_obj_list[1] = "dummy_shelf1"
                elif "bottom" in task_description:
                    task_related_obj_list[1] = "dummy_shelf0"

            move_obj_name, ref_obj_name, _ = task_related_obj_list
            print("task_related_obj_list: ", task_related_obj_list)


            # 初始化抓取相对位置
            T_o_g = None
            move_pose = get_abs_pose('gripper_pose', obs, task)
            ref_pose = get_pose_for_task(obs, task_name, task_description, move_obj_name, task)
            T_o_g = compute_relative_pose_T(move_pose, ref_pose,)


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
            move_pose = get_pose_for_task(obs, task_name, task_description, move_obj_name, task)
            ref_pose = get_abs_pose(ref_obj_name, obs, task)
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


                if smooth_idx == -1:
                    gripper_pose = obs.gripper_pose
                    gripper_pose[2] -= 0.0001
                    stage_change = torch.zeros(1).to(device).long()
                    print("init")
                else:
                    arm = noicy_action.detach().cpu().numpy().squeeze()
                    all_time_actions[smooth_idx][smooth_idx:smooth_idx+act_trunk] = arm
                    smooth_action = action_smooth(all_time_actions, smooth_idx)
                    stage_change = gripper_model(hist_obs, task_emb, state)
                    stage_change = stage_change.argmax().item()

                    gripper_pose = get_abs_action(smooth_action, ref_pose, rot_type, T_o_g)

                smooth_idx += 1


                action = np.concatenate([gripper_pose, gripper_state])
                obs, reward, done = task_env.step(action)

                if i > max_timesteps-20 and gripper_state == 0:
                    gripper_state = 1 - gripper_state

                # if stage_change:
                #     state += 1
                #     print(f"stage_change: {state}")
                #     break
                    # gripper_state = 1 - gripper_state

                # hist_update
                move_pose = get_pose_for_task(obs, task_name, task_description, move_obj_name, task)
                ref_pose = get_abs_pose(ref_obj_name, obs, task)
                relative_pose = compute_relative_pose_input(move_pose, ref_pose, rot_type).reshape(1, -1)

                hist_obs = torch.cat([
                    hist_obs[:, 1:],
                    torch.from_numpy(relative_pose).unsqueeze(0).to(device)
                ], dim=1).float()

                if done:
                    print("Episode success!")
                    break

            iters += 1  # 正常完成
            if done:
                success += 1  # 成功
                result_msg = f"Episode {iters} success!"
            else:
                result_msg = f"Episode {iters} failed."
            
            # 同时打印到终端和日志文件
            print(result_msg)
            with open(log_file, 'a') as f:
                f.write(result_msg + '\n')

        except Exception as e:
            error_msg = f"Error in episode {iters}: {str(e)}"
            print(error_msg)
            pass
        finally:
            final_msg = f"task_description: {task_description}"
            print(final_msg)
            with open(log_file, 'a') as f:
                f.write(final_msg + '\n')

    final_result = f"[Task: {task_name}] Success rate: {success}/{test_iters} = {success/test_iters:.2%}"
    print(final_result)
    with open(log_file, 'a') as f:
        f.write("\n" + final_result + "\n")
        f.write("-" * 50 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="RLBench Dataset Generator")
    parser.add_argument('--save_path', '-s', type=str, default='./video_save', help='Where to save the demos.')
    parser.add_argument('--ckpt_dir', '-c', type=str, help='checkpoint dir.')
    parser.add_argument('--tasks', nargs='*', default=['close_jar'], help='The tasks to collect. If empty, all tasks are collected.')
    parser.add_argument('--image_size', nargs=2, type=int, default=[128, 128], help='The size of the images to save.')
    parser.add_argument('--variations', type=int, default=1, help='Number of variations to collect per task. -1 for all.')
    parser.add_argument('--iters', type=int, default=1, )
    parser.add_argument('--log_dir', type=str, )
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

    task_files = [
        # "close_jar",
        # "insert_onto_square_peg",
        "light_bulb_in",
        # "put_money_in_safe",
        # "reach_and_drag",
        # "stack_wine",
    ]

    tasks = [task_file_to_task_class(t) for t in task_files]

    tz, bert = init_bert()

    # 初始化 Hydra
    with initialize(version_base="1.3", config_path=args.ckpt_dir):
        # 加载配置文件
        cfg = compose(config_name="config")

    cfg.arm_model_path = os.path.join(args.ckpt_dir, 'arm_model_best.ckpt')
    cfg.gripper_model_path = os.path.join(args.ckpt_dir, 'gripper_model_best.ckpt')
    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
        obs_config=ObservationConfig(),
        headless=True)
    rlbench_env.launch()
    for t in tasks:
        start(rlbench_env, t, args, cfg, tz, bert)

    print('Finish')
    rlbench_env.shutdown()


if __name__ == '__main__':
    main()