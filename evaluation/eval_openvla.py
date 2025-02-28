import numpy as np
import argparse
import os
from PIL import Image
from pathlib import Path
import json
import imageio
from datetime import datetime

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.backend.utils import task_file_to_task_class
import rlbench.backend.task as task

import torch
from pyrep.objects import Object
from coordiff.utils.transfrom import *
from coordiff.models import *
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.transform import Rotation as R


def quaternion_to_euler(quat, sequence='xyz', degrees=False):
    """
    将四元数转换为欧拉角。

    参数:
        quat (list): 四元数，形式为 [w, x, y, z]。
        sequence (str): 欧拉角的旋转顺序，例如 'xyz'、'zyx' 等。
        degrees (bool): 是否返回角度值（True）或弧度值（False，默认）。

    返回:
        list: 欧拉角列表，形式为 [roll, pitch, yaw]。
    """
    rotation = R.from_quat(quat)
    euler_angles = rotation.as_euler(sequence, degrees=degrees)
    return euler_angles.tolist()

def quat2rpy(pose):
    quat = pose[3:]
    quat = np.concatenate([[quat[3]],quat[:3]])
    euler = quaternion_to_euler(quat)
    return np.concatenate([pose[:3], euler])

class Openvla_Wrapper:
    def __init__(self, openvla_path):
        self.openvla_path = openvla_path
        self.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)

    def predict_action(self, obs, lang):
        front_rgb = Image.fromarray(obs.front_rgb)

        prompt = lang[0]

        inputs = self.processor(prompt, front_rgb).to(self.device, dtype=torch.bfloat16)
        action = self.vla.predict_action(**inputs, do_sample=False)
        def euler_to_quaternion_scipy(roll, pitch, yaw):
            """
            Convert Euler angles (roll, pitch, yaw) to a quaternion using SciPy.
            Angles must be in radians.
            Returns a quaternion as (w, x, y, z).
            """
            r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
            return r.as_quat()  # Ret
        quat = euler_to_quaternion_scipy(action[3], action[4],action[5])

        return np.concatenate([action[:3],quat[1:],[quat[0], action[6]]])  

def numpy_arrays_to_video(image_list, video_name, fps=10):
    """
    Save a list of NumPy arrays as a video.

    Parameters:
        image_list (list of np.ndarray): List of NumPy arrays representing images.
        video_name (str): Name of the output video file.
        fps (int): Frames per second for the video.
    """
    # Write the video
    with imageio.get_writer(video_name, fps=fps) as writer:
        for image in image_list:
            writer.append_data(image)



def eval_once(env, model):
    task_description, obs = env.reset()
    # print("task_description: ", task_description)
    success = False
    # front_rgb_list = []
    for i in tqdm(range(100)):
        try:
            action = model.predict_action(obs, task_description)
            obs, reward, done = env.step(action)
        except:
            break
        if done:
            success = True
            break
        # front_rgb_list.append(obs.front_rgb)
    return success
    # numpy_arrays_to_video(front_rgb_list, 'test.mp4')

def eval_task(task, model):
    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
        obs_config=ObservationConfig(),
        headless=False)
    rlbench_env.launch()
    task_env = rlbench_env.get_task(task)

    task_name = task_env.get_name()
    success_num = 0

    test_num = 50
    for i in range(test_num):
        variation_count = task_env.variation_count()
        if variation_count == 1:
            variation = 0
        else:
            variation = np.random.randint(0, variation_count-1)
        task_env.set_variation(variation)
        if eval_once(task_env, model):
            success_num += 1
        print(f'{task_name}  success rate: {success_num} / {i+1}')
    rlbench_env.shutdown()
    return success_num / test_num


def parse_args():
    parser = argparse.ArgumentParser(description="RLBench Dataset Generator")
    parser.add_argument('--save_path', '-s', type=str, default='./save', help='Where to save the demos.')
    parser.add_argument('--ckpt_dir', '-c', type=str, help='checkpoint dir.')
    parser.add_argument('--tasks', nargs='*', default=['light_bulb_in'], help='The tasks to collect. If empty, all tasks are collected.')
    parser.add_argument('--image_size', nargs=2, type=int, default=[128, 128], help='The size of the images to save.')
    parser.add_argument('--variations', type=int, default=1, help='Number of variations to collect per task. -1 for all.')
    return parser.parse_args()


def main():
    args = parse_args()

    model = Openvla_Wrapper('/home/user/openvla/log/openvla-7b+rl_bench+b10+lr-0.0005+lora-r32+dropout-0.0+rl_bench+b10+lr-0.0005+lora-r32+dropout-0.0')

    task_files = ['close_jar', 'insert_onto_square_peg', 'light_bulb_in', 'meat_off_grill', 'place_cups',            'place_shape_in_shape_sorter', 'put_groceries_in_cupboard',  
                  'put_money_in_safe', 'reach_and_drag', 'stack_blocks', 'stack_cups', 'stack_wine']
    task_files = args.tasks
    tasks = [task_file_to_task_class(t) for t in task_files]

    result = dict()
    for i, task in enumerate(tasks):
        success_rate = eval_task(task, model)
        result[task_files[i]] = success_rate

    save_dir = Path(f'result/openvla/{task_files[i]}')
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(save_dir, 'openvla.txt')   
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(file_path, 'w') as file:
        file.write(f'Timestamp: {current_time}\n\n')
        
        file.write('Successful Rate of Diffenent Task:\n')
        for key, value in result.items():
            file.write('\n')
            file.write(f'{key}: {value}')
        
    print(f'Data has been saved to {file_path}')
    print('Finish')


if __name__ == '__main__':
    main()