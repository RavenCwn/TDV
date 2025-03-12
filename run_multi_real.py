import os
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", "-d", 
                        default="./data")
    parser.add_argument("--config_name", "-c", 
                        default="coordiff")
    parser.add_argument("--work_dir", "-w", 
                        default=None)
    args = parser.parse_args()
    
    root_dir = args.root_dir

    CONFIG_NAME = args.config_name
    task_list = os.listdir(os.path.join(args.root_dir, 'train'))
    work_dir = "./results/coordiff_real"

    task_list = ["pour"]

    if args.work_dir is None:
        work_dir = os.path.join(work_dir, f"{time.strftime('%m-%d_%H-%M-%S')}")
    else:
        work_dir = os.path.join(work_dir, args.work_dir, f"{time.strftime('%m-%d_%H-%M-%S')}")

    os.makedirs(work_dir)

    for seed in range(1):
        commond = (f'python -m coordiff.engine.train_only --config-name={CONFIG_NAME}'
                    f' train_dataset="{root_dir}" val_dataset="{root_dir}"'
                    f' seed={seed}'
                    # f' +task_list={task_list}'
                    f' +task_list=[{",".join(task_list)}]'
                    f" +work_dir={work_dir}"
                    f" use_language=True"
                    f" batch_size=32"
                    # f" batch_size=1024"
                    f" num_states=5"
                    f" epochs=2001"
                    # f" +resume_path=results/coordiff_real/pour_test/03-01_03-46-50"
                    )
        os.system(commond)