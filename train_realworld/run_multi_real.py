import os
import argparse
import time
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", "-d", 
                        default="coordiff_real_world/recollection_final")
    parser.add_argument("--config_name", "-c", 
                        default="coordiff_mlp_rms -w coordiff_mlp_rms_leaf")
    parser.add_argument("--work_dir", "-w", 
                        default=None)
    parser.add_argument("--task_name1", "-t1", 
                        default='teapot_coaster_single_01234_360_d')
    parser.add_argument("--task_name2", "-t2", 
                        default=None)
    parser.add_argument("--hist_len", "-hl", 
                        default=5, type=int)
    parser.add_argument("--epochs", "-e", 
                        default=2001, type=int)
    parser.add_argument("--action_trunk", "-a", 
                        default=10, type=int)
    parser.add_argument("--batch_size",  
                        default=32, type=int)
    parser.add_argument('--use_language', action='store_true', default=False)
    
    args = parser.parse_args()
    
    root_dir = args.root_dir

    CONFIG_NAME = args.config_name
    # task_list = os.listdir(args.root_dir)
    work_dir = "./results/coordiff_real"

    # task_list = ["put_new_gen"] # 1. 修改任务名字
    if args.task_name2 == None:
        task_list = [args.task_name1] # 1. 修改任务名字
    else:
        task_list = [args.task_name1, args.task_name2]


    if args.work_dir is None:
        work_dir = os.path.join(work_dir, f"{time.strftime('%m-%d_%H-%M-%S')}")
    else:
        # work_dir = os.path.join(work_dir, args.work_dir, f"{time.strftime('%m-%d_%H-%M-%S')}")
        work_dir = os.path.join(work_dir, args.work_dir, f"{args.task_name1}")


    os.makedirs(work_dir, exist_ok=True)

    # # 如果原路径下手部路径存在，则复制到输出目录中
    T_o_hand_path = f"coordiff_real_world/recollection_final/train/{args.task_name1}/episode_0/0000"
    print('T_o_hand_path', T_o_hand_path)
    if os.path.exists(os.path.join(T_o_hand_path, f"T_o_hand_pose.npy")):
        shutil.copy(os.path.join(T_o_hand_path, f"T_o_hand_pose.npy"), os.path.join(work_dir, f"T_o_hand_pose.npy"))

    
    for task in task_list:
        for fname in ("pos_mean.npy", "pos_var.npy"):
            src = os.path.join(root_dir, 'train', task, 'episode_0/0000', fname)
            if os.path.isfile(src):
                dst = os.path.join(work_dir, fname)
                shutil.copy(src, dst)
                print(f"复制{src} → {dst}")


    for seed in range(1):
        commond = (f'python -m coordiff.engine.train_only --config-name={CONFIG_NAME}'
                    f' train_dataset="{root_dir}" val_dataset="{root_dir}"'
                    f' seed={seed}'
                    # f' +task_list={task_list}'
                    f' +task_list=[{",".join(task_list)}]'
                    f" +work_dir={work_dir}"
                    f" use_language={args.use_language}"
                    f" batch_size={args.batch_size}"
                    # f" batch_size=2048"
                    f" num_states=1"
                    f" epochs={args.epochs}"
                    f" act_trunk={args.action_trunk}"
                    f" hist_len={args.hist_len}"

                    # f" +resume_path=results/coordiff_real/pour_test/03-01_03-46-50"
                    )
        os.system(commond)