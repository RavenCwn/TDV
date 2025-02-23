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


def save(img, path, idx): 
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, f'{idx:04}.png')
    img.save(path) 
    return idx+1

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


def run_waypoint(waypoint, task_env):
    waypoint.start_of_path()
    if waypoint.skip:
        return
    path = waypoint.get_path()

    ext = waypoint.get_ext()
    path.visualize()

    done = False
    success = False
    gripper_open = 1
    while not done:
        done = path.step()
        task_env._scene.step()
        # task_env._scene._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
        success, term = task_env._task.success()
        obs = task_env._scene.get_observation()

        if success:
            break


    if len(ext) > 0:
        contains_param = False
        start_of_bracket = -1
        gripper = task_env._scene.robot.gripper
        if 'open_gripper(' in ext:
            gripper.release()
            start_of_bracket = ext.index('open_gripper(') + 13
            contains_param = ext[start_of_bracket] != ')'
            if not contains_param:
                done = False
                while not done:
                    gripper_open = 1.0
                    done = gripper.actuate(gripper_open, 0.04)
                    task_env._scene.step()
                    task_env._scene._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)

        elif 'close_gripper(' in ext:
            start_of_bracket = ext.index('close_gripper(') + 14
            contains_param = ext[start_of_bracket] != ')'
            if not contains_param:
                done = False
                while not done:
                    gripper_open = 0.0
                    done = gripper.actuate(gripper_open, 0.04)
                    task_env._scene.step()
                    task_env._scene._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)

        if contains_param:
            rest = ext[start_of_bracket:]
            num = float(rest[:rest.index(')')])
            done = False
            while not done:
                gripper_open = num
                done = gripper.actuate(gripper_open, 0.04)
                task_env._scene.step()
                task_env._scene._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)

        if 'close_gripper(' in ext:
            for g_obj in task_env._scene.task.get_graspable_objects():
                gripper.grasp(g_obj)

    waypoint.end_of_path()
    path.clear_visualization()

different_obj_list = {
    "put_groceries_in_cupboard": [
        'crackers',
        'chocolate jello',
        'strawberry jello',
        'soup',
        'tuna',
        'spam',
        'coffee',
        'mustard',
        'sugar',
    ],
    "meat_off_grill": ['chicken', 'steak'],
    "place_shape_in_shape_sorter": ['cube', 'cylinder', 'triangular prism', 'star', 'moon'],
}


def get_pose_for_task(obs, task_name, task_description, obj_name, task):
    target_obj = None
    # different 任务根据任务描述和物体名字获得参考物体的pose
    print("task_name: ", task_name)
    if task_name in list(different_obj_list.keys()):
        for obj in different_obj_list[task_name]:
            print(obj, task_description)
            if obj in task_description:
                target_obj = obj.replace(' ', '_')
                print("grasp_obj: ", target_obj)
                
                break
        ref_pose = get_abs_pose(target_obj, obs, task)
    else:
        target_obj = obj_name
        ref_pose = get_abs_pose(target_obj, obs, task)
        print("grasp_obj: ", target_obj)

    return ref_pose

def start(task, args, cfg, tz, bert):
    idx = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
        obs_config=ObservationConfig(),
        headless=False)
    rlbench_env.launch()

    task_env = rlbench_env.get_task(task)
    # task_env.set_variation(args.variations)
    task_description, obs = task_env.reset()
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
    task_name = task.get_name()
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
    DDIM.set_timesteps(32)
    DDIM.alphas_cumprod = (
        DDIM.alphas_cumprod.to(device)
    )

    max_timesteps = 200
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


    for i in tqdm(range(200)):

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
            print("arm: ", arm)
            smooth_action = action_smooth(all_time_actions, smooth_idx)
            stage_change = gripper_model(hist_obs, task_emb, state)
            stage_change = stage_change.argmax().item()

            print("T_o_g: ", T_o_g)

            # TODO 获得参考物体的pose
            # gripper_pose = get_abs_action(arm[0], ref_pose, rot_type, T_o_g)
            gripper_pose = get_abs_action(smooth_action, ref_pose, rot_type, T_o_g)
            print("smooth_action: ", smooth_action)
            print("gripper_pose: ", gripper_pose)
            print("ref_pose: ", ref_pose)

        smooth_idx += 1


        action = np.concatenate([gripper_pose, gripper_state])
        obs, reward, done = task_env.step(action)

        # if stage_change:
        #     state += 1
        #     print(f"stage_change: {state}")
        #     break
        #     # gripper_state = 1 - gripper_state

        #     if move_obj_name == 'gripper_pose':
        #         move_pose = get_abs_pose(move_obj_name, obs, task)
        #         ref_pose = get_abs_pose(ref_obj_name, obs, task)
        #         print("move_obj_name: ", move_obj_name)
        #         print("ref_obj_name: ", ref_obj_name)
        #         T_o_g = compute_relative_pose_T(move_pose, ref_pose,)
        #         print("T_o_g: ", T_o_g)

        #     move_obj_name, ref_obj_name = task_stage[state.item()]
        #     print("move_obj_name: ", move_obj_name)
        #     print("ref_obj_name: ", ref_obj_name)
        #     move_pose = get_abs_pose(move_obj_name, obs, task)
        #     ref_pose = get_abs_pose(ref_obj_name, obs, task)
        #     relative_pose = compute_relative_pose_input(move_pose, ref_pose, rot_type)

        #     hist_obs = [torch.from_numpy(relative_pose) for _ in range(hist_len)]
        #     hist_obs = torch.stack(hist_obs, dim=0).unsqueeze(0).to(device).float()

        front_rgb = Image.fromarray(obs.front_rgb)
        path = os.path.join(args.save_path, task.get_name(), 'front_rgb')
        idx = save(front_rgb, path, idx)
        
        # print("jar0 pose: ", get_abs_pose('jar0', obs, task))
        # print("jar1 pose: ", get_abs_pose('jar1', obs, task))

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
            # break
        step += 1

    rlbench_env.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description="RLBench Dataset Generator")
    parser.add_argument('--save_path', '-s', type=str, default='./save', help='Where to save the demos.')
    parser.add_argument('--ckpt_dir', '-c', type=str, help='checkpoint dir.')
    parser.add_argument('--tasks', nargs='*', default=['put_money_in_safe'], help='The tasks to collect. If empty, all tasks are collected.')
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

    tasks = [task_file_to_task_class(t) for t in task_files]

    tz, bert = init_bert()

    # 初始化 Hydra
    with initialize(version_base="1.3", config_path=args.ckpt_dir):
        # 加载配置文件
        cfg = compose(config_name="config")

    cfg.arm_model_path = os.path.join(args.ckpt_dir, 'arm_model_400.ckpt')
    cfg.gripper_model_path = os.path.join(args.ckpt_dir, 'gripper_model_400.ckpt')

    for t in tasks:
        start(t, args, cfg, tz, bert)

    print('Finish')


if __name__ == '__main__':
    main()