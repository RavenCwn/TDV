import os
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", "-d", 
                        default="./data")
    parser.add_argument("--config_name", "-c", 
                        default="coordiff_saperate")
    parser.add_argument("--work_dir", "-w", 
                        default=None)
    args = parser.parse_args()
    
    root_dir = args.root_dir

    CONFIG_NAME = args.config_name
    task_list = os.listdir(os.path.join(args.root_dir, 'train'))
    # print(task_list)
    # task_list = ['pour_water', ]
    # task_list = ['place_shape_in_shape_sorter']
    # task_list = ['light_bulb_in']
    task_list = ['stack_blocks']

    work_dir = "./results/coordiff_saperate"

    if args.work_dir is None:
        work_dir = os.path.join(work_dir, f"{time.strftime('%m-%d_%H-%M-%S')}")
    else:
        work_dir = os.path.join(work_dir, args.work_dir, f"{time.strftime('%m-%d_%H-%M-%S')}")

    os.makedirs(work_dir)

    for seed in range(1):
        commond = (f'python -m coordiff.engine.train_sp --config-name={CONFIG_NAME}'
                    f' train_dataset="{root_dir}" val_dataset="{root_dir}"'
                    f' seed={seed}'
                    # f' +task_list={task_list}'
                    f' +task_list=[{",".join(task_list)}]'
                    f" +work_dir={work_dir}"
                    f" use_language=True"
                    f" batch_size=256"
                    # f" batch_size=2048"
                    f" num_states=5"
                    f" epochs=2001"
                    f" +resume_path='results/coordiff_saperate/all_rms_sp/02-25_16-52-25'"
                    # f" +resume_path='results/coordiff/all_larger_dit/02-24_15-14-21'"
                    # f" +resume_path='results/coordiff/all_lunch4/02-23_14-13-28'"
                    # f" +resume_path='results/coordiff/all_place_shape_s/02-24_01-57-01'"
                    # f" +resume_path='results/coordiff/all_data_fix/02-22_16-45-47'"
                    )
        os.system(commond)