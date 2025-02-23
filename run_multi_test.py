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
    # print(task_list)
    task_list = ['close_jar', ]

    work_dir = "./results/coordiff"

    if args.work_dir is None:
        work_dir = os.path.join(work_dir, f"{time.strftime('%m-%d_%H-%M-%S')}")
    else:
        work_dir = os.path.join(work_dir, args.work_dir, f"{time.strftime('%m-%d_%H-%M-%S')}")

    os.makedirs(work_dir)

    for seed in range(1):
        commond = (f'python -m coordiff.engine.train2 --config-name={CONFIG_NAME}'
                    f' train_dataset="{root_dir}" val_dataset="{root_dir}"'
                    f' seed={seed}'
                    # f' +task_list={task_list}'
                    f' +task_list=[{",".join(task_list)}]'
                    f" +work_dir={work_dir}"
                    f" use_language=true"
                    f" batch_size=10240"
                    # f" +resume_path='/home/a4090/lfwh/CoorDiff/results/coordiff/close_jar_wo_language/02-20_10-12-20'"
                    
                    )
        os.system(commond)