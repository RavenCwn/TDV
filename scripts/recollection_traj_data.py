import os
import os.path as osp
import numpy as np
from glob import glob
from easydict import EasyDict
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import random
import shutil
import torch
import pytorch3d.transforms as pt3d
from coordiff.utils.transfrom import *




def init_bert():
    tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir="./bert")
    model = AutoModel.from_pretrained(
        "bert-base-cased", cache_dir="./bert")
    return tz, model

def get_task_embs(cfg, description, tz, model):
    """
    Bert embeddings for task embeddings. Borrow from https://github.com/Lifelong-Robot-Learning/LIBERO/blob/f78abd68ee283de9f9be3c8f7e2a9ad60246e95c/libero/lifelong/utils.py#L152.
    """
    
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


def complex_task(episode_path, output_path):
    pass

def simple_task(episode_path, output_path, task_name):
    '''
        这里的任务流程是固定的
    '''
    with open(osp.join(episode_path, 'language', 'variation_descriptions.txt'), 'r') as f:
        task_descriptions = f.readlines()
    print(task_descriptions[0].strip())

    ''' state level '''

    traj_path = osp.join(episode_path, 'trajectory')
    print("traj_path: ", traj_path)
    gripper_open_with_force = np.loadtxt(traj_path + '/gripper_open_with_force.txt', delimiter=' ')


    with open(osp.join("examples/task_stages", f'{task_name}.txt'), 'r') as f:
        task_related_obj_list = f.read().strip().split()

    last_gripper_state = 1
    state = 0
    start_idx = 0
    end_idx = 0
    re_state_path = None

    for step in range(gripper_open_with_force.shape[0]):
        gripper_state = gripper_open_with_force[step]

        if gripper_state == 0 and last_gripper_state == 1:
            print(f"Grasp obj start recording: {step}")
            start_idx = step
            re_state_path = os.makedirs(osp.join(output_path, f'{state:04d}'), exist_ok=True)

        elif gripper_state == 1 and last_gripper_state == 0:
            print(f"Release obj end recording: {step}")
            moving_obj, reference_obj = task_related_obj_list[:2]
            moving_traj = np.loadtxt(osp.join(traj_path, f"{moving_obj}.txt"), delimiter=' ')
            reference_traj = np.loadtxt(osp.join(traj_path, f"{reference_obj}.txt"), delimiter=' ')
            end_idx = step

            moving_traj = moving_traj[start_idx:end_idx]
            reference_traj = reference_traj[start_idx:end_idx]
            relevant_traj = compute_relative_traj_input(moving_traj, reference_traj, rot_type='6d')

            np.save(osp.join(re_state_path, 'relevant_traj.npy'), relevant_traj)
            task_description = task_descriptions[0].strip()
            task_embs = get_task_embs(cfg, task_description, tz, model).cpu().numpy()
            np.save(osp.join(re_state_path, 'task_description.npy'), task_description)
            np.save(osp.join(re_state_path, 'task_language_embed.npy'), task_embs)
            gripper_change = np.zeros(relevant_traj.shape[0])
            gripper_change[-1] = 1
            np.save(osp.join(re_state_path, 'gripper_change.npy'), gripper_change)

            state = 0
            start_idx = 0
            end_idx = 0
            re_state_path = None
            state += 1

        last_gripper_state = gripper_state
    

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

def different_task(episode_path, output_path, task_name):
    '''
        这里需要在
    '''
    with open(osp.join(episode_path, 'language', 'variation_descriptions.txt'), 'r') as f:
        task_descriptions = f.readlines()
    print(task_descriptions[0].strip())
    task_description = task_descriptions[0].strip()

    ''' state level '''
    traj_path = osp.join(episode_path, 'trajectory')
    print("traj_path: ", traj_path)
    gripper_open_with_force = np.loadtxt(traj_path + '/gripper_open_with_force.txt', delimiter=' ')


    with open(osp.join("examples/task_stages", f'{task_name}.txt'), 'r') as f:
        task_related_obj_list = f.read().strip().split()

    last_gripper_state = 1
    state = 0
    start_idx = 0
    end_idx = 0
    re_state_path = None

    for step in range(gripper_open_with_force.shape[0]):
        gripper_state = gripper_open_with_force[step]

        if gripper_state == 0 and last_gripper_state == 1:
            print(f"Grasp obj start recording: {step}")
            start_idx = step
            re_state_path = os.makedirs(osp.join(output_path, f'{state:04d}'), exist_ok=True)

        elif gripper_state == 1 and last_gripper_state == 0:
            print(f"Release obj end recording: {step}")
            moving_obj, reference_obj = task_related_obj_list[:2]

            for obj in different_obj_list[task_name]:
                if obj in task_description:
                    moving_obj = obj
                    break

            assert moving_obj == 'object', "Occupant error."

            moving_traj = np.loadtxt(osp.join(traj_path, f"{moving_obj}.txt"), delimiter=' ')
            reference_traj = np.loadtxt(osp.join(traj_path, f"{reference_obj}.txt"), delimiter=' ')
            end_idx = step

            moving_traj = moving_traj[start_idx:end_idx]
            reference_traj = reference_traj[start_idx:end_idx]
            relevant_traj = compute_relative_traj_input(moving_traj, reference_traj, rot_type='6d')

            np.save(osp.join(re_state_path, 'relevant_traj.npy'), relevant_traj)
            task_embs = get_task_embs(cfg, task_description, tz, model).cpu().numpy()
            np.save(osp.join(re_state_path, 'task_description.npy'), task_description)
            np.save(osp.join(re_state_path, 'task_language_embed.npy'), task_embs)
            gripper_change = np.zeros(relevant_traj.shape[0])
            gripper_change[-1] = 1
            np.save(osp.join(re_state_path, 'gripper_change.npy'), gripper_change)

            state = 0
            start_idx = 0
            end_idx = 0
            re_state_path = None
            state += 1

        last_gripper_state = gripper_state
    



if __name__ == '__main__':
    '''
    将原数据转换成 coordiff 需要的数据
    recollect_traj_data
        task_name
            episode_xxxx
                state_xxxx
                    relevant_traj.npy
                    task_description.npy
                    task_language_embed.npy
                    gripper_change.npy
    '''

    root_dir = "./combine_var_data"
    recollection_dir = "./recollect_traj_data"
    task_name = sorted(os.listdir(root_dir))
    print(task_name)

    ''' init bert model '''
    tz, model = init_bert()

    cfg = EasyDict({
        "task_embedding_format": "bert",
        "task_embedding_one_hot_offset": 1,
        "data": {"max_word_len": 25},
        "policy": {"language_encoder": {"network_kwargs": {"input_size": 768}}}
    })  # hardcode the config to get task embeddings according to original Libero code

        
    simple_task_list = [
        "close_jar",
        "insert_onto_square_peg",
        "light_bulb_in",
        "put_money_in_safe",
        "reach_and_drag",
        "stack_wine",
        "stack_cups"
    ]

    different_obj_task_list = [
        "meat_off_grill",
        "place_shape_in_shape_sorter",
        "put_groceries_in_cupboard",
    ]

    complex_task_list = [
        "place_cups",
        # "stack_blocks",
        "turn_tap"
    ]

    ''' task level '''
    for t in tqdm(task_name):
        task_path = osp.join(root_dir, t)
        re_task_path = osp.join(recollection_dir, t)
        os.makedirs(re_task_path, exist_ok=True)

        ''' episode level '''
        episodes_list = sorted(os.listdir(task_path))
        for e in episodes_list:
            print(f"Task: {t}, Episode: {e}")
            episode_path = osp.join(task_path, e)
            re_episode_path = osp.join(re_task_path, e)
            os.makedirs(re_episode_path, exist_ok=True)

            if task_name in simple_task_list:
                simple_task(episode_path, re_episode_path, task_name)
            elif task_name in different_obj_task_list:
                different_task(episode_path, re_episode_path, task_name)
            elif task_name in complex_task_list:
                complex_task(episode_path, re_episode_path, task_name)
            else:
                raise ValueError("Unknown task name")


         



