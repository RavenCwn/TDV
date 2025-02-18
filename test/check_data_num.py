from glob import glob
import os
import natsort
import shutil

# task_dirs = sorted(glob("all_data/*"))

# print(task_dirs)

task_dirs = {
    'all_data/close_jar': 7,
    'all_data/insert_onto_square_peg': 7,
    'all_data/light_bulb_in': 7,
    'all_data/meat_off_grill': 75,
    'all_data/place_cups': 42,
    'all_data/place_shape_in_shape_sorter': 25,
    'all_data/put_groceries_in_cupboard': 16,
    'all_data/put_money_in_safe': 42,
    'all_data/reach_and_drag': 7,
    'all_data/stack_blocks': 21,
    'all_data/stack_cups': 7,
    'all_data/stack_wine': 125,
    'all_data/turn_tap': 75
}

keys = list(task_dirs.keys())
for i in range(len(keys)):
    idx = i
    num_epi = task_dirs[keys[idx]]

    task_dir = keys[idx]  # Ensure task_dirs is defined earlier

    var_dirs = os.listdir(task_dir)
    tmp_dir = "other_data"

    for var_dir in var_dirs:
        episode_dir = os.path.join(task_dir, var_dir, 'episodes')
        if not os.path.exists(episode_dir):
            continue  # Skip if no episodes directory exists

        epi_dirs = natsort.natsorted(os.listdir(episode_dir))  # Ensure sorted order
        assert len(epi_dirs) == num_epi, f"Expected {num_epi} episodes in {episode_dir}, but got {len(epi_dirs)}"
        # for i, epi in enumerate(epi_dirs):
            # if i >= num_epi:
            #     # Move excess training data to another folder
            #     epi_path = os.path.join(episode_dir, epi)
            #     other_path = os.path.join(tmp_dir, task_dir, var_dir, 'episodes', epi)

            #     # Ensure the destination directory exists
            #     os.makedirs(os.path.dirname(other_path), exist_ok=True)

            #     # Move the file or folder
            #     shutil.move(epi_path, other_path)
            #     print(f"Moved {epi_path} -> {other_path}")