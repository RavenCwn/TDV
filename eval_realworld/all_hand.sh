task_name=${1}

hand_select="right"
obj_name='hand'



ref_name='socket'
ref_prompt='socket'

# ref_name='drawer_lllast'
# ref_prompt='drawer'

# ref_name='tissue'
# ref_prompt='tissue box'

batch_size=8

source /home/user/miniconda3/bin/activate   # 替代conda init命令
# 第零步：检查数据


# # 第一部：识别手部坐标(将数据放到WILOR的数据目录下)
ln -s /home/user/pgp/CoorDiff/coordiff_real_world/object_image_collector /home/user/ljc/WiLoR/datasets
# cp -r coordiff_real_world/object_image_collector/${task_name} /home/user/ljc/WiLoR/datasets
conda activate hamer
cd /home/user/ljc/WiLoR

bash process_auto.sh datasets/${task_name} ${hand_select} 0       # 此处默认只会执行episode_0的手部识别
# python hand_trajectory_collection_mediapipe.py -i datasets/${task_name}/episode_0

cp -r datasets/${task_name} /home/user/ljc/AutoPoseEstimator/Data/data_real


# # # # # 第二部：识别参考物体坐标
cd /home/user/ljc/AutoPoseEstimator
conda activate auto_pose
# 注意这次只跟踪第一帧的
python run_simulate_camera_GD.py --only_first --name ${ref_name} --text_prompt "${ref_prompt}" --task ${task_name} 
# python run_simulate_camera_GD.py --name ${ref_name} --text_prompt "${ref_prompt}" --task ${task_name} 


# # 第三步：提取参考物体和手的初始相对坐标并保存
cd /home/user/pgp/CoorDiff
conda activate coordiff
python calculate_o_hand_auto.py --T1_path coordiff_real_world/final_data/${task_name}/episode_0/${ref_name}_poses.npy --T2_path coordiff_real_world/final_data/${task_name}/episode_0/hand_poses.npy --output_path coordiff_real_world/final_data/${task_name}/episode_0/T_o_hand_pose.npy

# # 第四步：计算参考物体和手的相对坐标
cd coordiff_real_world
python process_relevant.py -t ${task_name} -r ${ref_name} -o ${obj_name}

# 第五步：开始训练
cd /home/user/pgp/CoorDiff
conda activate coordiff
python run_multi_real.py --batch_size ${batch_size} -d coordiff_real_world/recollection_final -c coordiff_mlp_rms -w coordiff_mlp_rms_leaf -t1 ${task_name}


# 第六步：正式运行(运行前先开位姿检测)
# python deploy_hand_auto.py  -c results/coordiff_real/coordiff_mlp_rms_leaf/cwn_drawer