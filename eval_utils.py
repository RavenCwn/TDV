import os
import numpy as np
import torch
from easydict import EasyDict
from transformers import AutoTokenizer, AutoModel
from pyrep.objects import Object
from coordiff.utils.transform import *
from coordiff.models import *

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

from pyrep.errors import ConfigurationPathError
from pyrep.const import ObjectType
from rlbench.backend.exceptions import (
    WaypointError, BoundaryError, NoWaypointsError, DemoError)
def run_waypoint(waypoint, task_env):
    waypoint.start_of_path()
    if waypoint.skip:
        return

    grasped_objects = task_env._scene.robot.gripper.get_grasped_objects()
    colliding_shapes = [s for s in task_env._scene.pyrep.get_objects_in_tree(
        object_type=ObjectType.SHAPE) if s not in grasped_objects
                        and s not in task_env._scene._robot_shapes and s.is_collidable()
                        and task_env._scene.robot.arm.check_arm_collision(s)]
    [s.set_collidable(False) for s in colliding_shapes]
    try:
        path = waypoint.get_path()
        [s.set_collidable(True) for s in colliding_shapes]
    except ConfigurationPathError as e:
        [s.set_collidable(True) for s in colliding_shapes]
        raise DemoError(
            'Could not get a path for waypoint',
            task_env._scene.task) from e
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
    # print("task_name: ", task_name)
    if task_name in list(different_obj_list.keys()):
        for obj in different_obj_list[task_name]:
            # print(obj, task_description)
            if obj in task_description:
                target_obj = obj.replace(' ', '_')
                # print("grasp_obj: ", target_obj)
                
                break
        ref_pose = get_abs_pose(target_obj, obs, task)
    else:
        target_obj = obj_name
        ref_pose = get_abs_pose(target_obj, obs, task)
        # print("grasp_obj: ", target_obj)

    return ref_pose