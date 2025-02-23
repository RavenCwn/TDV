import numpy as np
import argparse
import os
import os.path as osp
from PIL import Image

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.backend.utils import task_file_to_task_class

import rlbench.backend.task as task
import torch
from coordiff.utils.transform import *
from coordiff.models import *
from hydra import initialize, compose
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm
import shutil
from eval_utils import *

complex_obj_list = {
    "place_cups": [
        "place 1 cup on the cup holder",
        "place 2 cups on the cup holder",
        "place 3 cups on the cup holder",
    ],
    "stack_blocks": [
        'stack %d %s blocks' % (2, "red"),
        'stack %d %s blocks' % (3, "red"),
        'stack %d %s blocks' % (4, "red"),
        'stack %d %s blocks' % (2, "maroon"),
        'stack %d %s blocks' % (3, "maroon"),
        'stack %d %s blocks' % (4, "maroon")
    ],
    "turn_tap": [
        "turn left tap",
        "turn right tap"
    ],
    "stack_cups": []
}

def start(task, args, cfg, tz, bert):
    idx = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
        obs_config=ObservationConfig(),
        headless=False)
    rlbench_env.launch()

    task_env = rlbench_env.get_task(task)
    task_env.set_variation(2)
    task_description, obs = task_env.reset()
    print(f"task_description: {task_description}")
    task_description = task_description[0]
    
    waypoints = task_env._task.get_waypoints()

    waypoint_idx = 0
    for i, point in enumerate(waypoints[waypoint_idx:waypoint_idx+3]):
        run_waypoint(point, task_env)
        obs = task_env.get_observation()
    # waypoint_idx += 3

    # while True:
    #     success = False
    #     for i, point in enumerate(waypoints):
    #         print(i)
    #         run_waypoint(point, task_env)
    #         obs = task_env.get_observation()
    #         input()
    #         success, term = task_env._task.success()

    #     if not task_env._task.should_repeat_waypoints() or success:
    #         break

    task = task_env._task
    scene = task_env._scene
    robot = task_env._robot
    random_seed = np.random.get_state()
    task_name = task.get_name()
    hist_len = cfg.hist_len
    rot_type = cfg.rot_type


    id = -1
    for i, task_desc in enumerate(complex_obj_list[task_name]):
        if task_desc == task_description:
            id = i % 3
            break
    
    task_stages_path = None
    if task_name == "stack_cups":
        task_stages_path = osp.join("examples/task_stages", f'{task_name}.txt')
    else:
        task_stages_path = osp.join("examples/task_stages", f'{task_name}_{id}.txt')
    with open(task_stages_path, 'r') as f:
        task_related_obj_list = f.readlines()
        task_related_obj_list = [obj.strip().split() for obj in task_related_obj_list]
    num_stages = len(task_related_obj_list)

    state = torch.tensor([0]).to(device).long()
    move_obj_name, ref_obj_name = task_related_obj_list[state.item()]
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
    DDIM.alphas_cumprod = (
        DDIM.alphas_cumprod.to(device)
    )

    max_timesteps = 400
    act_trunk = arm_model.act_trunk
    input_dim = arm_model.input_dim
    all_time_actions = np.zeros([max_timesteps, max_timesteps+act_trunk, input_dim])


    # 初始化观测
    # TODO 获得参考物体的pose 和 move obj pose
    move_pose = get_pose_for_task(obs, task_name, task_description, move_obj_name, task)
    ref_pose = get_abs_pose(ref_obj_name, obs, task)
    relative_pose = compute_relative_pose_input(move_pose, ref_pose, rot_type)
    print("relative_pose: ", relative_pose)

    # hist_obs = [torch.from_numpy(relative_pose) for _ in range(hist_len)]
    # hist_obs = torch.stack(hist_obs, dim=0).unsqueeze(0).to(device).float()
    hist_obs = torch.zeros([hist_len, input_dim]).unsqueeze(0).to(device).float()
    hist_obs[-1] = torch.from_numpy(relative_pose).to(device).float()
    task_emb = get_task_embs(cfg, task_description, tz, bert).to(device).float()
    gripper_state = np.array([0])
    stage_change = torch.zeros(1).to(device).long()


    step = 0
    done = False
    smooth_idx = -1

    path = os.path.join(args.save_path, task.get_name(), 'front_rgb')
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)

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
            # gripper_pose = get_abs_action(arm[5], ref_pose, rot_type, T_o_g)
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

            # print("T_o_g: ", T_o_g)

            # TODO 获得参考物体的pose
            # gripper_pose = get_abs_action(arm[0], ref_pose, rot_type, T_o_g)
            gripper_pose = get_abs_action(smooth_action, ref_pose, rot_type, T_o_g)
            # print("smooth_action: ", smooth_action)
            # print("gripper_pose: ", gripper_pose)
            # print("ref_pose: ", ref_pose)

        smooth_idx += 1


        action = np.concatenate([gripper_pose, gripper_state])
        obs, reward, done = task_env.step(action)


        if state < num_stages and stage_change:
            state += 1
            print(f"stage_change: {state}")
            
            gripper_state = np.array([1])
            gripper_pose = obs.gripper_pose
            gripper_pose[2] -= 0.0001
            action = np.concatenate([gripper_pose, gripper_state])
            obs, reward, done = task_env.step(action)
            
            if state < num_stages:
                run_waypoint(waypoints[-1], task_env)
                obs = task_env.get_observation()
                # input()
                print("last step")
                task_env._task.should_repeat_waypoints()
                print("start planning")
                for i, point in enumerate(waypoints[waypoint_idx:waypoint_idx+3]):
                    # print(i)
                    # input()
                    run_waypoint(point, task_env)
                    obs = task_env.get_observation()
                # waypoint_idx += 3

                move_obj_name, ref_obj_name = task_related_obj_list[state.item()]
                move_pose = get_abs_pose('gripper_pose', obs, task)
                ref_pose = get_pose_for_task(obs, task_name, task_description, move_obj_name, task)
                T_o_g = compute_relative_pose_T(move_pose, ref_pose,)
                gripper_state = np.array([0])

                move_obj_name, ref_obj_name = task_related_obj_list[state.item()]
                print("move_obj_name: ", move_obj_name)
                print("ref_obj_name: ", ref_obj_name)
                move_pose = get_abs_pose(move_obj_name, obs, task)
                ref_pose = get_abs_pose(ref_obj_name, obs, task)
                relative_pose = compute_relative_pose_input(move_pose, ref_pose, rot_type)

                all_time_actions = np.zeros([max_timesteps, max_timesteps+act_trunk, input_dim])
                smooth_idx = -1

            # hist_obs = [torch.from_numpy(relative_pose) for _ in range(hist_len)]
            # hist_obs = torch.stack(hist_obs, dim=0).unsqueeze(0).to(device).float()
            hist_obs = torch.zeros([hist_len, input_dim]).unsqueeze(0).to(device).float()
            hist_obs[-1] = torch.from_numpy(relative_pose).to(device).float()
        elif stage_change:
            gripper_state = np.array([1])

        front_rgb = Image.fromarray(obs.front_rgb)
        idx = save(front_rgb, path, idx)
        
        # hist_update
        # 获得运动物体和参考物体的pose
        # move_pose = get_abs_pose(move_obj_name, obs, task)
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
        step += 1

    rlbench_env.shutdown()
    print(f"task_description: {task_description}")


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
    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    if len(args.tasks) > 0:
        for t in args.tasks:
            if t not in task_files:
                raise ValueError('Task %s not recognised!.' % t)
        task_files = args.tasks

    task_files = ["stack_blocks"]
    complex_task_list = [
        "place_cups",
        "stack_blocks",
        "stack_cups",
    ]
    tasks = [task_file_to_task_class(t) for t in task_files]

    tz, bert = init_bert()

    # 初始化 Hydra
    with initialize(version_base="1.3", config_path=args.ckpt_dir):
        # 加载配置文件
        cfg = compose(config_name="config")

    cfg.arm_model_path = os.path.join(args.ckpt_dir, 'arm_model_best.ckpt')
    cfg.gripper_model_path = os.path.join(args.ckpt_dir, 'gripper_model_best.ckpt')

    for t in tasks:
        start(t, args, cfg, tz, bert)

    print('Finish')


if __name__ == '__main__':
    main()