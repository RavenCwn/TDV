import numpy as np
from glob import glob
import os
import os.path as osp
import torch
from torch.utils.data import Dataset, DataLoader
import random

def get_dataloader(replay, mode, num_workers, batch_size):
    loader = DataLoader(
        replay,
        shuffle=(mode == "train"),
        pin_memory=False,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None
    )
    return loader

class TrajDataset(Dataset):
    def __init__(self,
                dataset_dir,
                act_chunk,
                hist_len,
                mode,
                rot_type='6d',
                task_list=None,
    ):
        super().__init__()

        assert mode in ['train', 'val'], "mode must be 'train' or 'val'"

        self.dataset_dir = dataset_dir
        self.mode = mode
        if task_list is None:  # default to all tasks
            self.task_list = os.listdir(dataset_dir)
        else:
            self.task_list = task_list
        
        self.act_chunk = act_chunk
        self.hist_len = hist_len
        
        episode_states = []
        for task in self.task_list:
            task_path = osp.join(dataset_dir, task)
            task_path = osp.join(task_path, mode)
            # print(task_path)
            
            # 同样的任务具有同样的states
            episodes = sorted(glob(osp.join(task_path, "*")))
            states = os.listdir(episodes[0])
            # print(episodes)

            for episode in episodes:
                for state in states:
                    # if state == '1':
                    episode_states.append((episode, state))
        random.shuffle(episode_states)

        self._index_to_demo_id = {}
        self._demo_id_to_start_indices = {}
        self._demo_id_to_demo_length = {}

        self.relevant_trajs = []
        self.gripper_changes = []
        self.task_embs = []
        self.actions = []
        self.state = []
        start_idx = 0
        demo_idx = 0
        for episode_state in episode_states:
            demo_len, relevant_traj, gripper_change, task_emb, action, state = self.load_demo(episode_state)

            self.relevant_trajs.append(relevant_traj)
            self.gripper_changes.append(gripper_change)
            self.task_embs.append(task_emb)
            self.actions.append(action)
            self.state.append(state)

            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            start_idx += demo_len
            demo_idx += 1

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx

    def load_demo(self, episode_state):
        epi, state = episode_state
        relevant_traj = np.load(osp.join(epi, state, "relevant_traj.npy"))
        gripper_change = np.load(osp.join(epi, state, "gripper_change.npy"))
        task_emb = np.load(osp.join(epi, state, "task_de_embed.npy")).squeeze()
        # print(gripper_change.shape, relevant_traj.shape, task_emb.shape)

        demo_len = relevant_traj.shape[0]

        relevant_traj = np.concatenate([relevant_traj, np.repeat(relevant_traj[-1:], self.hist_len, axis=0)], axis=0)
        gripper_change = np.concatenate([gripper_change[self.hist_len:], np.repeat(gripper_change[-1:], self.hist_len, axis=0)], axis=0)  # no chunk
        action = np.concatenate([relevant_traj[self.hist_len:], np.repeat(relevant_traj[-1:], self.act_chunk, axis=0)], axis=0)
        # print(gripper_change.shape, relevant_traj.shape, action.shape)

        return demo_len, relevant_traj, gripper_change, task_emb, action, np.array(float(state))


    def __getitem__(self, index):
        # print("index: ", index)
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = index - demo_start_index
        # print("time_offset: ", time_offset)

        # print("relevant_traj: ", time_offset, time_offset + self.hist_len, len(self.relevant_trajs), demo_id)
        
        relevant_traj = self.relevant_trajs[demo_id][time_offset:time_offset + self.hist_len]
        
        
        gripper_change = np.array(self.gripper_changes[demo_id][time_offset])
        task_emb = self.task_embs[demo_id]
        action = self.actions[demo_id][time_offset:time_offset + self.act_chunk]
        state = self.state[demo_id]
        
        relevant_traj = torch.from_numpy(relevant_traj).float()
        gripper_change = torch.from_numpy(gripper_change).long()
        task_emb = torch.from_numpy(task_emb).float()
        action = torch.from_numpy(action).float()
        state = torch.from_numpy(state).long()

        return relevant_traj, action, task_emb, state, gripper_change

    def __len__(self):
        return len(self._index_to_demo_id)


if __name__ == "__main__":
    dataset = TrajDataset(
        dataset_dir="./data",
        act_chunk=15,
        hist_len=10,
        mode='train'
    )
    print(len(dataset))
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, data in enumerate(dataloader):
        print(i)
        relevant_traj, action, task_emb, state, gripper_change = data
        print(relevant_traj.shape, action.shape, task_emb.shape, state.shape, gripper_change.shape)