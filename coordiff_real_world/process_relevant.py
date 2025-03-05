from glob import glob
import numpy as np
from coordiff.utils.transform import *
import os
import os.path as osp
from transformers import AutoTokenizer, AutoModel
from easydict import EasyDict


def init_bert():
    tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir="../bert")
    model = AutoModel.from_pretrained(
        "bert-base-cased", cache_dir="../bert")
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


if __name__ == "__main__":

    origin_data_dir = "final_data"
    output_data_dir = "recollection_final/train"

    task_name_dirs = sorted(os.listdir(origin_data_dir))
    task_name_dirs = ["leaf8"]

    ''' init bert model '''
    tz, model = init_bert()

    cfg = EasyDict({
        "task_embedding_format": "bert",
        "task_embedding_one_hot_offset": 1,
        "data": {"max_word_len": 25},
        "policy": {"language_encoder": {"network_kwargs": {"input_size": 768}}}
    })  # hardcode the config to get task embeddings according to original Libero code

    for task_name in task_name_dirs:
        task_dir = os.path.join(origin_data_dir, task_name)
        episode_dirs = sorted(os.listdir(task_dir))
        task_description = task_name

        for epi in episode_dirs:
            epi_dir = os.path.join(task_dir, epi)
            for state in range(1):
                ref_obj = np.load(f"{epi_dir}/teajar_poses.npy")
                move_obj = np.load(f"{epi_dir}/teaspoon_poses.npy")

                print(f"ref_obj: {ref_obj.shape}, move_obj: {move_obj.shape}")

                relevant_obj = np.linalg.inv(ref_obj) @ move_obj
                print(f"relevant_obj: {relevant_obj.shape}")

                ref_pos = ref_obj[:, :3, 3]
                move_pos = move_obj[:, :3, 3]
                ref_quat = rotation_matrix_to_quaternion(ref_obj[:, :3, :3])
                move_quat = rotation_matrix_to_quaternion(move_obj[:, :3, :3])
                ref_pose = np.concatenate([ref_pos, ref_quat], axis=1)
                move_pose = np.concatenate([move_pos, move_quat], axis=1)
                
                print(ref_pose.shape, move_pose.shape)
                relevant_traj = compute_relative_traj_smooth(move_pose, ref_pose, rot_type='6d', smoothing_window=5)
                
                output_task_epi = os.path.join(output_data_dir, task_description, epi, f"{state:04d}")
                os.makedirs(output_task_epi, exist_ok=True)


                np.save(os.path.join(output_task_epi, 'relevant_traj.npy'), relevant_traj)
                
                task_embs = get_task_embs(cfg, task_description, tz, model).cpu().numpy()
                np.save(osp.join(output_task_epi, 'task_description.npy'), task_description)
                np.save(osp.join(output_task_epi, 'task_language_embed.npy'), task_embs)
                
                gripper_change = np.zeros(relevant_traj.shape[0])
                gripper_change[-1] = 1
                np.save(osp.join(output_task_epi, 'gripper_change.npy'), gripper_change)
                
                with open(osp.join(output_task_epi, 'task_related_objs.txt'), 'w') as f:
                    f.write(f"cup1 cup0")