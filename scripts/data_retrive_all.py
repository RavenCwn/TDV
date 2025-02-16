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


joint_positions_list = []
joint_velocities_list = []
joint_forces_list = []
gripper_open_list = []
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
    gripper_pose_list.clear()
    gripper_joint_positions_list.clear()
    gripper_touch_forces_list.clear()
    save_list = {}


def save_data(state, output_folder, task_stages):

    if len(joint_positions_list) < 10:
        clean_up()
        return
    # assert len(joint_positions_list) > 20, f"stage len too shot: {len(joint_positions_list)}"

    output_folder = os.path.join(output_folder, str(state))
    check_and_make(output_folder)
    with open(os.path.join(output_folder, 'task_related_obj.txt'), 'w') as file: 
        stages = task_stages[state]
        file.write(' '.join(stages))

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

task_dir = sorted(glob("all_data/*"))
output_dir = "retrive_data"

task_dir = ["all_data/close_jar"]
task_dir = ["all_data/reach_and_drag"]
task_dir = ["all_data/insert_onto_square_peg"]
task_dir = ["all_data/meat_off_grill"]
task_dir = ["all_data/stack_wine"]
task_dir = ["all_data/put_groceries_in_cupboard"]
task_dir = ["all_data/put_money_in_safe"]


# # 以下四个task可能有点问题
# task_dir = ["all_data/turn_tap"]
# task_dir = ["all_data/stack_blocks"]
# task_dir = ["all_data/stack_cups"]
# task_dir = ["all_data/place_cups"]

    


for task in task_dir:
    task_name = task.split('/')[-1]
    origin_path = task + "/variation0"
    output_path = os.path.join(output_dir, task_name)

    with open(os.path.join("examples/task_stages", task_name, "stages.txt")) as file:
        task_stages = file.readlines()
    task_stages = [x.strip().split(" ") for x in task_stages]

    print(task_stages)

    for i, f in enumerate(ns.natsorted(os.listdir(origin_path + '/episodes'))):
        print(f)

        folder_path = os.path.join(origin_path + '/episodes', f)
        file_path = os.path.join(folder_path, 'low_dim_obs.pkl')
        out_pth = os.path.join(output_path, f'episode_{i:04d}')

        state = 0
        grasp = False

        with open(file_path, 'rb') as pkl_file:
            demo = pickle.load(pkl_file)

        for obs in demo:    
            if not grasp and (np.linalg.norm((obs.gripper_touch_forces)[:3]) > 0.1):
                save_data(state, out_pth, task_stages)
                state += 1
                grasp = True
            # elif grasp and (np.linalg.norm((obs.gripper_touch_forces)[:3]) < 0.01):
            #     save_data(state, out_pth, task_stages)
            #     state += 1
            #     if state > len(task_stages):
            #         break
            #     grasp = False

            joint_positions_list.extend([obs.joint_positions])
            joint_velocities_list.extend([obs.joint_velocities])
            joint_forces_list.extend([obs.joint_forces])
            gripper_open_list.extend([obs.gripper_open])
            gripper_pose_list.extend([obs.gripper_pose])
            gripper_joint_positions_list.extend([obs.gripper_joint_positions])
            gripper_touch_forces_list.extend([obs.gripper_touch_forces])

            for obj_name in obs.objs_pose.keys():  # 逐步按7个数据一组处理
                if obj_name not in save_list:  # Check if obj_name is not in save_list
                    save_list[obj_name] = []  # Initialize an empty list for obj_name
                save_list[obj_name].append(obs.objs_pose[obj_name])
        
        save_data(state, out_pth, task_stages)
        
        file2_path = origin_path + '/variation_descriptions.pkl'
        with open(file2_path, 'rb') as pkl_file:
            descriptions_list = pickle.load(pkl_file)
        
        path = os.path.join(out_pth, 'language')
        check_and_make(path)
        with open(os.path.join(path, 'descriptions.txt'), "w", encoding="utf-8") as f:
            for line in descriptions_list:
                f.write(line + "\n") 
                
        print('success process task: ', task_name)
