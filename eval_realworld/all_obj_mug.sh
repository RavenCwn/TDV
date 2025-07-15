task_name=${1}
model=${2}
batch_size=${3}
hand_select="right"

obj_name='bluemug'
obj_prompt='blue mug with yellow lining in the right'
ref_name='pinkmug'
ref_prompt='mug with pink color'

output_name="${task_name}_${model}" 
batch_size=${batch_size}

# ref_name=${2}
# obj_name=${3}


source /home/user/miniconda3/bin/activate   # 替代conda init命令
# 第零步：检查数据


# # # # 第一部：识别手部坐标(将数据放到WILOR的数据目录下)
ln -s /home/user/pgp/CoorDiff/coordiff_real_world/object_image_collector /home/user/ljc/WiLoR/datasets
# cp -r coordiff_real_world/object_image_collector/${task_name} /home/user/ljc/WiLoR/datasets
# conda activate hamer
cd /home/user/ljc/WiLoR
# bash process_auto.sh datasets/${task_name} ${hand_select} 0       # 此处默认只会执行episode_0的手部识别
cp -r datasets/${task_name} /home/user/ljc/AutoPoseEstimator/Data/data_real


# # 第二部：识别物体坐标
cd /home/user/ljc/AutoPoseEstimator
conda activate auto_pose
# 这里直接就保存到Coordiff文件夹了，不用复制
python run_simulate_camera_GD.py --name ${obj_name} --text_prompt "${obj_prompt}" --task ${task_name}
python run_simulate_camera_GD.py --name ${ref_name} --text_prompt "${ref_prompt}" --task ${task_name}


#(生成轨迹)#
cd /home/user/ljc/AutoPoseEstimator
conda activate auto_pose
# python gen_pose_move_scale_auto_dilute.py --need_rot --num_samples 100 --task ${task_name} --obj_name ${obj_name} --ref_name ${ref_name} --output_name ${output_name}
python gen_pose_move_scale_auto_dilute.py --model "${model}" --num_samples 100 --task ${task_name} --obj_name ${obj_name} --ref_name ${ref_name} --output_name ${output_name}
task_name=${output_name}


# # 第三步：提取物体和手的相对坐标
# cd /home/user/pgp/CoorDiff
# conda activate coordiff
# python calculate_o_hand_auto.py --T1_path coordiff_real_world/final_data/${task_name}/episode_0/${obj_name}_poses.npy --T2_path coordiff_real_world/final_data/${task_name}/episode_0/hand_poses.npy --output_path coordiff_real_world/final_data/${task_name}/episode_0/T_o_hand_pose.npy


# 第四步：计算物体和参考物体的相对坐标
task_name=${output_name}
cd /home/user/pgp/CoorDiff/coordiff_real_world
conda activate coordiff
python process_relevant.py -t ${task_name} -r ${ref_name} -o ${obj_name}


# # 第五步：开始训练
cd ..
python run_multi_real.py --batch_size ${batch_size} -d coordiff_real_world/recollection_final -c coordiff_mlp_rms -w coordiff_mlp_rms_leaf -t1 ${task_name}

# 第六步：补充相对坐标
cd results/coordiff_real/coordiff_mlp_rms_leaf
cp a_mug_T/T_o_hand_pose.npy ${output_name}

# 第六步：正式运行(运行前先开位姿检测)
# python deploy_silky_auto.py  -c results/coordiff_real/coordiff_mlp_rms_leaf/final_mug_pipeline_test







