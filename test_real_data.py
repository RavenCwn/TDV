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



def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_timesteps = 200
    num_stages = 1

    
    relevant_traj = np.load("/home/user/pgp/CoorDiff/coordiff_real_world/recollection_data_smooth5/train/pour3/episode_4/0000/relevant_traj.npy")
    task_emb = np.load("/home/user/pgp/CoorDiff/coordiff_real_world/recollection_data_smooth5/train/pour3/episode_4/0000/task_language_embed.npy")
    task_emb = torch.from_numpy(task_emb).float().to(device)

    
    ref_idx = 0
    relative_pose = relevant_traj[ref_idx]
    ref_idx += 1


    done = False
    smooth_idx = 0

    generated_trajectory = []
    gt_trajectory = []
    for ref_T in relevant_traj:
        pos = ref_T[:3]
        rot = pt3d.rotation_6d_to_matrix(
            torch.from_numpy(ref_T[3:]).unsqueeze(0)
        ).numpy()
        gt_trajectory.append((pos, rot))

    generated_trajectory = None
    plot_stepwise_trajectory(generated_trajectory, gt_trajectory)



if __name__ == '__main__':
    test()