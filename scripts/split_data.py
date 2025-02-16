import os
import random
import shutil
import os.path as osp
from glob import glob

recollection_dir = "./recollect_data"

if __name__ == '__main__':

    task_list = os.listdir(recollection_dir)
    # split_ratio = 0.8
    split_ratio = 100
    print(task_list)
    for task in task_list:
        task_dir = osp.join(recollection_dir, task)
        episodes_list = glob(osp.join(task_dir, "*"))
        num_len = len(episodes_list)
        random.shuffle(episodes_list)
        os.makedirs(osp.join(task_dir, "train"), exist_ok=True)
        os.makedirs(osp.join(task_dir, "val"), exist_ok=True)

        train_episodes = episodes_list[:split_ratio]
        val_episodes = episodes_list[split_ratio:]

        # 移动训练集文件
        for episode in train_episodes:
            shutil.move(episode, osp.join(task_dir, "train", osp.basename(episode)))
        
        # 移动测试集文件
        for episode in val_episodes:
            shutil.move(episode, osp.join(task_dir, "val", osp.basename(episode)))
