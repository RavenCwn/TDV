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


def start(task, args, cfg, tz, bert):
    idx = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
        obs_config=ObservationConfig(),
        headless=False)
    rlbench_env.launch()

    task_env = rlbench_env.get_task(task)
    task_env.set_variation(args.variations)
    task_description, obs = task_env.reset()
    print(f"task_description: {task_description}")
    
    
    waypoints = task_env._task.get_waypoints()
    for i, point in enumerate(waypoints[:2]):
       run_waypoint(point, task_env)

    task = task_env._task
    scene = task_env._scene
    robot = task_env._robot
    random_seed = np.random.get_state()
    
    hist_len = cfg.hist_len
    rot_type = cfg.rot_type


    state = torch.tensor([0]).to(device).long()
    with open(f"examples/task_stages/{task.get_name()}.txt", "r") as f:
        task_stage = f.readlines()
        task_stage = [x.strip().split(' ') for x in task_stage]
    print("task_stage: ", task_stage)


    # TODO 获得参考物体的pose 和 move obj pose
    move_obj_name, ref_obj_name = task_stage[state.item()]
    move_pose = get_abs_pose(move_obj_name, obs, task)
    ref_pose = get_abs_pose(ref_obj_name, obs, task)
    relative_pose = compute_relative_pose_input(move_pose, ref_pose, rot_type)



    done = False
    T_o_g = None
    move_pose = get_abs_pose('gripper_pose', obs, task)
    ref_pose = get_abs_pose(task_stage[0][0], obs, task)
    T_o_g = compute_relative_pose_T(move_pose, ref_pose,)


    step = 0

    arm_model, gripper_model = load_policy(cfg, device)
    DDIM = DDIMScheduler(**cfg.ddim_cfg)
    DDIM.set_timesteps(cfg.eval_timesteps)
    DDIM.alphas_cumprod = (
        DDIM.alphas_cumprod.to(device)
    )


    act_trunk = arm_model.act_trunk
    input_dim = arm_model.input_dim
    max_timesteps = 200
    all_time_actions = np.zeros([max_timesteps, max_timesteps+act_trunk, input_dim])

    smooth_idx = -1

    # hist_obs = [torch.from_numpy(relative_pose) for _ in range(hist_len)]
    # hist_obs = torch.stack(hist_obs, dim=0).unsqueeze(0).to(device).float()
    hist_obs = torch.zeros([hist_len, input_dim]).unsqueeze(0).to(device).float()
    hist_obs[-1] = torch.from_numpy(relative_pose).to(device).float()
    task_emb = get_task_embs(cfg, task_description[0], tz, bert).to(device).float()
    gripper_state = np.array([0])
    stage_change = torch.zeros(1).to(device).long()

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
            smooth_action = action_smooth(all_time_actions, smooth_idx)
            stage_change = gripper_model(hist_obs, task_emb, state)
            stage_change = stage_change.argmax().item()

            # TODO 获得参考物体的pose
            # gripper_pose = get_abs_action(arm[0], ref_pose, rot_type, T_o_g)
            gripper_pose = get_abs_action(smooth_action, ref_pose, rot_type, T_o_g)
        smooth_idx += 1


        action = np.concatenate([gripper_pose, gripper_state])
        obs, reward, done = task_env.step(action)

        if stage_change:
            state += 1
            print(f"stage_change: {state}")
            break
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
        
        # hist_update
        # 获得运动物体和参考物体的pose
        move_pose = get_abs_pose(move_obj_name, obs, task)
        ref_pose = get_abs_pose(ref_obj_name, obs, task)
        relative_pose = compute_relative_pose_input(move_pose, ref_pose, rot_type).reshape(1, -1)

        hist_obs = torch.cat([
            hist_obs[:, 1:],
            torch.from_numpy(relative_pose).unsqueeze(0).to(device)
        ], dim=1).float()

        if done:
            print("Episode success!")
        step += 1

    rlbench_env.shutdown()



def parse_args():
    parser = argparse.ArgumentParser(description="RLBench Dataset Generator")
    parser.add_argument('--save_path', '-s', type=str, default='./save', help='Where to save the demos.')
    parser.add_argument('--ckpt_dir', '-c', type=str, help='checkpoint dir.')
    parser.add_argument('--tasks', nargs='*', default=['insert_onto_square_peg'], help='The tasks to collect. If empty, all tasks are collected.')
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

    cfg.arm_model_path = os.path.join(args.ckpt_dir, 'arm_model_1000.ckpt')
    cfg.gripper_model_path = os.path.join(args.ckpt_dir, 'gripper_model_1000.ckpt')

    for t in tasks:
        start(t, args, cfg, tz, bert)

    print('Finish')


if __name__ == '__main__':
    main()