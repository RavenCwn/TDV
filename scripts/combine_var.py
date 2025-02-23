import pickle
import numpy as np
import os
from glob import glob
import natsort as ns
from rlbench import Environment
from rlbench import RandomizeEvery
from rlbench import VisualRandomizationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import ReachAndDrag
from rlbench.tasks import CloseJar

current_dir = os.path.dirname(os.path.abspath(__file__)) + '/../'

joint_positions_list = []
joint_velocities_list = []
joint_forces_list = []
gripper_open_list = []
gripper_open_with_force_list = []
gripper_pose_list = []
gripper_joint_positions_list = []
gripper_touch_forces_list = []
save_list = {} 


def check_and_make(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def clean_up():
    global save_list
    joint_positions_list.clear()
    joint_velocities_list.clear()
    joint_forces_list.clear()
    gripper_open_list.clear()
    gripper_open_with_force_list.clear()
    gripper_pose_list.clear()
    gripper_joint_positions_list.clear()
    gripper_touch_forces_list.clear()
    save_list = {}


def save_data(output_folder):

    if len(joint_positions_list) < 10:
        clean_up()
        return
    # assert len(joint_positions_list) > 20, f"stage len too shot: {len(joint_positions_list)}"

    check_and_make(output_folder)

    output_folder = os.path.join(output_folder, 'trajectory')
    check_and_make(output_folder)
    
    joint_positions_array = np.array(joint_positions_list)
    np.savetxt(os.path.join(output_folder, 'joint_positions.txt'), joint_positions_array, delimiter=' ')
    
    joint_velocities_array = np.array(joint_velocities_list)
    np.savetxt(os.path.join(output_folder, 'joint_velocities.txt'), joint_velocities_array, delimiter=' ')
    
    joint_forces_array = np.array(joint_forces_list)
    np.savetxt(os.path.join(output_folder, 'joint_forces.txt'), joint_forces_array, delimiter=' ')

    gripper_open_array = np.array(gripper_open_list)
    np.savetxt(os.path.join(output_folder, 'gripper_open.txt'), gripper_open_array, delimiter=' ')
    
    gripper_open_with_force_array = np.array(gripper_open_with_force_list)
    np.savetxt(os.path.join(output_folder, 'gripper_open_with_force.txt'), gripper_open_with_force_array, delimiter=' ')
    
    gripper_pose_array = np.array(gripper_pose_list)
    np.savetxt(os.path.join(output_folder, 'gripper_pose.txt'), gripper_pose_array, delimiter=' ')
    
    gripper_joint_positions_array = np.array(gripper_joint_positions_list)
    np.savetxt(os.path.join(output_folder, 'gripper_joint_positions.txt'), gripper_joint_positions_array, delimiter=' ')
    
    gripper_touch_forces_array = np.array(gripper_touch_forces_list)
    np.savetxt(os.path.join(output_folder, 'gripper_touch_forces.txt'), gripper_touch_forces_array, delimiter=' ')
    

    output_folder = os.path.join(output_folder, 'task_shape')
    check_and_make(output_folder)

    # 保存每个列表数据到对应的文件
    for i, group_data in save_list.items():
        group_data_array =  np.array(group_data)
        np.savetxt(os.path.join(output_folder, i + '.txt'), group_data_array, delimiter=' ')
    
    clean_up()


if __name__ == "__main__":
    '''
    将原数据转换成

    combine_var_data
        task_name
            episode_xxxx
                'left_shoulder_mask', # 软连接
                'overhead_mask',
                'overhead_depth',
                'wrist_depth',
                'front_mask',
                'left_shoulder_rgb',
                'right_shoulder_depth',
                'left_shoulder_depth',
                'wrist_rgb',
                'wrist_mask',
                'right_shoulder_rgb',
                'overhead_rgb',
                'right_shoulder_mask',
                'front_rgb',
                'front_depth'

                language
                    variation_descriptions.txt
                trajectory
                    joint_positions.txt
                    joint_velocities.txt
                    joint_forces.txt
                    gripper_open.txt
                    gripper_open_with_force.txt
                    gripper_pose.txt
                    gripper_joint_positions.txt
                    gripper_touch_forces.txt
                task_shape
                    obj_name.txt
                    ...
    '''

    # task_dir = sorted(glob("all_data/*"))
    # output_dir = "combine_var_data"

    # task_dir = ["all_data/close_jar"]
    # task_dir = ["all_data/reach_and_drag"]
    # task_dir = ["all_data/insert_onto_square_peg"]
    # task_dir = ["all_data/meat_off_grill"]
    # task_dir = ["all_data/stack_wine"]
    # task_dir = ["all_data/put_groceries_in_cupboard"]
    # task_dir = ["all_data/put_money_in_safe"]


    # # 以下四个task可能有点问题
    # task_dir = ["all_data/turn_tap"]
    # task_dir = ["all_data/stack_blocks"]
    # task_dir = ["all_data/stack_cups"]
    # task_dir = ["all_data/place_cups"]

    image_list = [
        'left_shoulder_mask',
        'overhead_mask',
        'overhead_depth',
        'wrist_depth',
        'front_mask',
        'left_shoulder_rgb',
        'right_shoulder_depth',
        'left_shoulder_depth',
        'wrist_rgb',
        'wrist_mask',
        'right_shoulder_rgb',
        'overhead_rgb',
        'right_shoulder_mask',
        'front_rgb',
        'front_depth'
    ]

    # num_epi_each_task = 100
    # mode = "train"
    
    num_epi_each_task = 25
    mode = "val"

    task_dir = sorted(glob("color_data/*"))
    output_dir = f"color_combine_var_data/{mode}"
    print(task_dir)
    for task in task_dir:
        task_name = task.split('/')[-1]
        origin_path = task
        variations = ns.natsorted(os.listdir(origin_path))
        num_var = len(variations)
        
        # num_epi_of_var = int(np.ceil(num_epi_each_task / num_var))
        num_epi_of_var = num_epi_each_task // num_var
        rest = num_epi_each_task - num_epi_of_var * num_var
        var_ids = [num_epi_of_var for _ in range(num_var)]
        var_ids[-1] += rest

        print(var_ids)

        total_idx = 0
        for v, var in enumerate(variations):
            var_path = os.path.join(task, var)
            var_descriptions = os.path.join(var_path, 'variation_descriptions.pkl')
            with open(var_descriptions, 'rb') as pkl_file:
                descriptions_list = pickle.load(pkl_file)

            output_path = os.path.join(output_dir, task_name)  # no var
            check_and_make(output_path)

            episode_paths = ns.natsorted(glob(var_path + '/episodes/*'))

            for i, f in enumerate(episode_paths[:var_ids[v]]):
                print(f"process [task:{task_name}]-[{var}]: {i}/{num_epi_of_var}")

                folder_path = f

                low_dim_file_path = os.path.join(folder_path, 'low_dim_obs.pkl')
                out_episode_path = os.path.join(output_path, f'episode_{total_idx:04d}')
                out_episode_img_path = os.path.join(output_path, f'episode_{total_idx:04d}/images')
                if os.path.exists(out_episode_path):
                    total_idx+=1
                    continue
                check_and_make(out_episode_path)
                check_and_make(out_episode_img_path)
                total_idx+=1

                for name in image_list:
                    origin_img_path = os.path.join(folder_path, name)
                    output_img_path = os.path.join(out_episode_img_path, name)
                    # print("origin_img_path: ", os.path.join(current_dir, origin_img_path))
                    # print("output_img_path: ", output_img_path)
                    os.symlink(os.path.join(current_dir, origin_img_path), output_img_path, target_is_directory=True)

                path = os.path.join(out_episode_path, 'language')
                check_and_make(path)
                with open(os.path.join(path, 'variation_descriptions.txt'), "w", encoding="utf-8") as f:
                    for line in descriptions_list:
                        f.write(line + "\n") 


                with open(low_dim_file_path, 'rb') as pkl_file:
                    demo = pickle.load(pkl_file)

                for obs in demo:
                    joint_positions_list.extend([obs.joint_positions])
                    joint_velocities_list.extend([obs.joint_velocities])
                    joint_forces_list.extend([obs.joint_forces])
                    gripper_open_list.extend([obs.gripper_open])
                    gripper_open_with_force_list.extend([np.linalg.norm((obs.gripper_touch_forces)[:3]) < 0.01])  # 1是张开 0是闭合
                    gripper_pose_list.extend([obs.gripper_pose])
                    gripper_joint_positions_list.extend([obs.gripper_joint_positions])
                    gripper_touch_forces_list.extend([obs.gripper_touch_forces])

                    for obj_name in obs.task_objects.keys():  # 逐步按7个数据一组处理
                        if obj_name not in save_list:  # Check if obj_name is not in save_list
                            save_list[obj_name] = []  # Initialize an empty list for obj_name
                        save_list[obj_name].append(obs.task_objects[obj_name])
                        assert len(save_list[obj_name]) == len(gripper_open_with_force_list)

                save_data(out_episode_path)
            
                        
        print('success process task: ', task_name)
