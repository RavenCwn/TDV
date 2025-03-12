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
from coordiff.utils.transform import *
import argparse

colors = [
    'red',
    'maroon',
    'lime',
    'green',
    'blue',
    'navy',
    'yellow',
    'cyan',
    'magenta',
    'silver',
    'gray',
    'orange',
    'olive',
    'purple',
    'teal',
    'azure',
    'violet',
    'rose',
    'black',
    'white',
]

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

def save_when_close(traj_path, moving_obj, reference_obj, start_idx, end_idx, re_state_path, task_description, tz, model):
    if moving_obj == "gripper_pose":
        moving_traj = np.loadtxt(osp.join(traj_path, f"{moving_obj}.txt"), delimiter=' ')
    else:
        moving_traj = np.loadtxt(osp.join(traj_path, f"task_shape/{moving_obj}.txt"), delimiter=' ')

    reference_traj = np.loadtxt(osp.join(traj_path, f"task_shape/{reference_obj}.txt"), delimiter=' ')

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
    with open(osp.join(re_state_path, 'task_related_objs.txt'), 'w') as f:
        f.write(f"{moving_obj} {reference_obj}")



def save_stages(gripper_open_with_force, task_related_obj_list, traj_path, output_path, task_description, tz, model):
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
            re_state_path = osp.join(output_path, f'{state:04d}')
            os.makedirs(re_state_path, exist_ok=True)

        elif gripper_state == 1 and last_gripper_state == 0:
            print(f"Release obj end recording: {step}")
            moving_obj, reference_obj = task_related_obj_list[state][:2]
            print("moving_obj, reference_obj: ", moving_obj, reference_obj)
            end_idx = step
            if end_idx - start_idx > 10:
                save_when_close(traj_path, moving_obj, reference_obj, start_idx, end_idx, re_state_path, task_description, tz, model)
                state += 1
           
            start_idx = 0
            end_idx = 0
            re_state_path = None

        last_gripper_state = gripper_state

    if re_state_path is not None:
        print(f"Release obj end recording: {step}")
        if state == len(task_related_obj_list):
            print("state: ", state)
            import ipdb; ipdb.set_trace()
            return
        moving_obj, reference_obj = task_related_obj_list[state][:2]

        end_idx = step
        if end_idx - start_idx > 10:
            save_when_close(traj_path, moving_obj, reference_obj, start_idx, end_idx, re_state_path, task_description, tz, model)
            state += 1

    if state == 0 or state >= 3:
        print("state: ", state)
        import ipdb; ipdb.set_trace()

def complex_task(episode_path, output_path, task_name):

    with open(osp.join(episode_path, 'language', 'variation_descriptions.txt'), 'r') as f:
        task_descriptions = f.readlines()
    print(task_descriptions[0].strip())
    task_description = task_descriptions[0].strip()

    traj_path = osp.join(episode_path, 'trajectory')
    print("traj_path: ", traj_path)
    gripper_open_with_force = np.loadtxt(traj_path + '/gripper_open_with_force.txt', delimiter=' ')

    id = -1
    for i, task_desc in enumerate(complex_obj_list[task_name]):
        if task_desc == task_description:
            id = i % 3
            break
    
    task_stages_path = None
    if task_name == "stack_cups":
        task_stages_path = osp.join("examples/task_stages", f'{task_name}.txt')
        with open(task_stages_path, 'r') as f:
            task_related_obj_list = f.readlines()
            task_related_obj_list = [obj.strip().split() for obj in task_related_obj_list]
        color_obj_list = os.listdir(osp.join(traj_path, f"task_shape"))
        for obj in color_obj_list:
            for task_related_obj in task_related_obj_list:
                if task_related_obj[0] == obj.split('-')[0]:
                    task_related_obj[0] = obj.removesuffix(".txt")
                if task_related_obj[1] == obj.split('-')[0]:
                    task_related_obj[1] = obj.removesuffix(".txt")
    else:
        task_stages_path = osp.join("examples/task_stages", f'{task_name}_{id}.txt')
        with open(task_stages_path, 'r') as f:
            task_related_obj_list = f.readlines()
            task_related_obj_list = [obj.strip().split() for obj in task_related_obj_list]

    save_stages(gripper_open_with_force, task_related_obj_list, traj_path, output_path, task_description, tz, model)
   
def simple_task(episode_path, output_path, task_name):
    '''
        这里的任务流程是固定的
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
        task_related_obj_list = f.readlines()
        task_related_obj_list = [obj.strip().split() for obj in task_related_obj_list]

    if task_related_obj_list[0][2] != "None":
        target_color = None
        for color in colors:
            if color in task_description:
                target_color = color
                break
        assert target_color is not None, f"Select color error. {task_description}, {color}"

        idx = 0 if task_related_obj_list[0][2] == 'move_obj' else 1
        target_name = task_related_obj_list[0][idx]
        color_obj_list = os.listdir(osp.join(traj_path, f"task_shape"))
        for obj in color_obj_list:
            if (target_color in obj) and (target_name in obj):
                task_related_obj_list[0][idx] = obj.removesuffix(".txt")
                target_color = 'None'
                target_name = 'None'
            elif task_related_obj_list[0][1-idx] == obj.split('-')[0]:
                task_related_obj_list[0][1-idx] = obj.removesuffix(".txt")
                
        assert task_related_obj_list[0][idx] is not None, f"Select obj error. {task_description}, {os.listdir(osp.join(traj_path, f'task_shape'))}"

    if "light_bulb_in" in episode_path:
        target_bulb_holder = np.loadtxt(osp.join(traj_path, f"task_shape/{task_related_obj_list[0][idx]}.txt"), delimiter=' ')[0][:3]
        for obj in color_obj_list:
            if "bulb0" in obj:
                bulb0 = (np.loadtxt(osp.join(traj_path, f"task_shape/{obj}"), delimiter=' ')[0][:3], obj.removesuffix(".txt"))
            elif "bulb1" in obj:
                bulb1 = (np.loadtxt(osp.join(traj_path, f"task_shape/{obj}"), delimiter=' ')[0][:3], obj.removesuffix(".txt"))
        if np.linalg.norm(bulb0[0] - target_bulb_holder) < np.linalg.norm(bulb1[0] - target_bulb_holder):
            task_related_obj_list[0][idx] = bulb0[1]
        else:
            task_related_obj_list[0][idx] = bulb1[1]

    if "put_money_in_safe" in episode_path:
        if "top" in task_description:
            task_related_obj_list[0][1] = "dummy_shelf2"
        elif "middle" in task_description:
            task_related_obj_list[0][1] = "dummy_shelf1"
        elif "bottom" in task_description:
            task_related_obj_list[0][1] = "dummy_shelf0"

        assert "dummy_shelf" in task_related_obj_list[0][1], f"Select obj error. {task_description}, {os.listdir(osp.join(traj_path, f'task_shape'))}"

    print(task_related_obj_list)
    save_stages(gripper_open_with_force, task_related_obj_list, traj_path, output_path, task_description, tz, model)
   
   

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
        这里需要根据任务描述确定操作物体
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
            re_state_path = osp.join(output_path, f'{state:04d}')
            os.makedirs(re_state_path, exist_ok=True)

        elif gripper_state == 1 and last_gripper_state == 0:
            print(f"Release obj end recording: {step}")
            moving_obj, reference_obj = task_related_obj_list[:2]

            for obj in different_obj_list[task_name]:
                if obj in task_description:
                    moving_obj = obj
                    break

            moving_obj = moving_obj.replace(' ', '_')
            if task_name == "place_shape_in_shape_sorter":
                temp = moving_obj
                # moving_obj = temp + '_grasp_point'
                reference_obj = temp + '_drop_point'
            # elif task_name == "put_groceries_in_cupboard":
            #     moving_obj += '_grasp_point'

            assert moving_obj != 'object', f"Occupant error. {task_description}, {different_obj_list[task_name]}"

            end_idx = step
            if end_idx - start_idx > 10:
                save_when_close(traj_path, moving_obj, reference_obj, start_idx, end_idx, re_state_path, task_description, tz, model)
                state += 1

            start_idx = 0
            end_idx = 0
            re_state_path = None

        last_gripper_state = gripper_state

    if re_state_path is not None:
        print(f"Release obj end recording: {step}")
        moving_obj, reference_obj = task_related_obj_list[:2]

        for obj in different_obj_list[task_name]:
            if obj in task_description:
                moving_obj = obj
                break

        moving_obj = moving_obj.replace(' ', '_')
        assert moving_obj != 'object', f"Occupant error. {task_description}, {different_obj_list[task_name]}"
        
        end_idx = step
        if end_idx - start_idx > 10:
            save_when_close(traj_path, moving_obj, reference_obj, start_idx, end_idx, re_state_path, task_description, tz, model)
            state += 1

    if state == 0 or state >= 3:
        print("state: ", state)
        import ipdb; ipdb.set_trace()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", required=True, type=str, help="train or val")
    args = parser.parse_args()

    mode = args
    # mode = "val"
    # mode = "train"

    root_dir = f"data/combine_var_data/{mode}"
    recollection_dir = f"data/recollect_traj_data/{mode}"
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
        # "close_jar",
        # "insert_onto_square_peg",
        # "light_bulb_in",
        # "put_money_in_safe",
        # "reach_and_drag",
        # "stack_wine",
        # "turn_tap"
    ]

    different_obj_task_list = [
        # "meat_off_grill",
        # "place_shape_in_shape_sorter",
        # "put_groceries_in_cupboard",
    ]

    complex_task_list = [
        # "place_cups",
        "stack_blocks",
        # "stack_cups",
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
            if osp.exists(re_episode_path):
                continue
            os.makedirs(re_episode_path, exist_ok=True)

            if t in simple_task_list:
                simple_task(episode_path, re_episode_path, t)
            elif t in different_obj_task_list:
                different_task(episode_path, re_episode_path, t)
            elif t in complex_task_list:
                complex_task(episode_path, re_episode_path, t)
            else:
                raise ValueError(f"Unknown task name {t}")


         



