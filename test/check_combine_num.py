import os
from tqdm import tqdm

root_dir = "./color_combine_var_data/train"
task_name = sorted(os.listdir(root_dir))
print(task_name)

num_epi_each_task = 100


for t in tqdm(task_name):
    task_path = os.path.join(root_dir, t)
    epi_dirs = sorted(os.listdir(task_path))
    
    assert len(epi_dirs) == num_epi_each_task, f"Expected {num_epi_each_task} episodes in {task_path}, but got {len(epi_dirs)}"