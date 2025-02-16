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

root_dir = "./retrive_data"
recollection_dir = "./recollect_data"
task_name = sorted(os.listdir(root_dir))
print(task_name)


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






if __name__ == '__main__':

    ''' init bert model '''
    tz, model = init_bert()

    cfg = EasyDict({
        "task_embedding_format": "bert",
        "task_embedding_one_hot_offset": 1,
        "data": {"max_word_len": 25},
        "policy": {"language_encoder": {"network_kwargs": {"input_size": 768}}}
    })  # hardcode the config to get task embeddings according to original Libero code

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


            with open(osp.join(episode_path, 'language', 'descriptions.txt'), 'r') as f:
                task_descriptions = f.readlines()
            print(task_descriptions[0].strip())

            ''' state level '''
            state_list = sorted(os.listdir(episode_path))
            state_list.remove("language")
            for s in state_list:
                state_path = osp.join(episode_path, s)
                print("\nstate_path: ", state_path)
                re_state_path = osp.join(re_episode_path, s)
                os.makedirs(re_state_path, exist_ok=True)


                with open(osp.join(state_path, 'task_related_obj.txt'), 'r') as f:
                    task_related_obj_list = f.read().strip().split()
                print(f"Task related object list: {task_related_obj_list}")
                

                traj_path = osp.join(state_path, 'trajectory')
                print("traj_path: ", traj_path)
                
                task_shape_list = sorted(glob(osp.join(traj_path, 'task_shape', '*.txt')))
                gripper_list = sorted(glob(osp.join(traj_path, '*.txt')))
                all_list = task_shape_list + gripper_list

                moving_obj, reference_obj = task_related_obj_list[:2]
                moving_traj, reference_traj = 0, 0

                print("moving / reference: ", moving_obj, reference_obj)
                for o in all_list:
                    o_name = o.split(".txt")[0].split("/")[-1]
                    if moving_obj == o_name:
                        assert moving_traj == 0, "Moving object trajectory already exists"
                        moving_traj = np.loadtxt(o, delimiter=' ')
                        print(f"Moving object: {o}")

                    if reference_obj == o_name:
                        assert reference_traj == 0, "Reference object trajectory already exists"
                        reference_traj = np.loadtxt(o, delimiter=' ')
                        print(f"Reference object: {o}")

                relevant_traj = compute_relative_traj_input(moving_traj, reference_traj, rot_type='6d')
                np.save(osp.join(re_state_path, 'relevant_traj.npy'), relevant_traj)
                task_description = task_descriptions[0].strip()
                task_embs = get_task_embs(cfg, task_description, tz, model).cpu().numpy()
                np.save(osp.join(re_state_path, 'task_description.npy'), task_description)
                np.save(osp.join(re_state_path, 'task_de_embed.npy'), task_embs)
                gripper_change = np.zeros(relevant_traj.shape[0])
                gripper_change[-1] = 1
                np.save(osp.join(re_state_path, 'gripper_change.npy'), gripper_change)



