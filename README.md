# TbTd: Trajectory-based Transformer Diffusion

该工作主要是完成基于轨迹的diffusion model

## 1. 环境安装
```
conda create -n coordiff python=3.9
conda activate coordiff
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install pytorch3d -c pytorch3d
```
可能遇到直接跳过的情况，这是因为 defaults 源没有 forvec 的依赖，需要换成清华源
```
conda config --remove-key channels
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/
conda config --set show_channel_urls true
```

然后按照rlbench的安装需要安装rlbench的依赖和库

## 2. 收集数据
```
# 这里一定要用自己的库，相对于官方的做出了部分修改
git clone git@github.com:inFpZero/RLbench_Coordiff.git

bash collected_data.sh
```

## 3. 训练
配置 coordiff 环境
```
git clone git@github.com:GuoPingPan/CoorDiff.git
git checkout -b swanlab
pip install -r requirements.txt 
pip install -e .
```
然后将数据放到 `CoorDiff/data` 文件夹下面
```
# 解析数据
python scripts/combine_var.py -m train
python scripts/combine_var.py -m val

python scripts/recollection_traj_data.py -m train
python scripts/recollection_traj_data.py -m val
```


训练仿真数据
```
python run_multi.py -d data_path -c model_config_name -w save_dir  # 相同 diffusion 步长的arm model 和 gripper model
python run_multi_sp.py -d data_path -c model_config_name -w save_dir # 不同 diffusion 步长的arm model 和 gripper model

# example
python train/run_multi.py -d data/color3_recollect_traj_data -c coordiff_mlp_rms -w coordiff_mlp_rms_all_task  # 相同 diffusion 步长的arm model 和 gripper model
```

训练真实数据
```
python train/run_multi_real.py -d coordiff_real_world/data_real/leaf8 -c coordiff_mlp_rms -w coordiff_mlp_rms_leaf  # 相同 diffusion 步长的arm model 和 gripper model
```
python train/run_multi_real.py -d coordiff_real_world/recollection_final -c coordiff_mlp_rms -w coordiff_mlp_rms_leaf -t teapot_coaster_single_01234_rottest

# 4. 验证

仿真验证
```
bash bash_eval.sh
```

真实验证：需要利用 ur5 和 6DOF Pose Estimation模块
```
python deploy.py -c results/coordiff_real/lip8/03-02_03-22-38
```
python deploy.py -c results/coordiff_real/coordiff_mlp_rms_leaf/04-02_22-35-07
python deploy.py -c results/coordiff_real/coordiff_mlp_rms_leaf/04-01_16-29-44
python deploy.py  -c results/coordiff_real/coordiff_mlp_rms_leaf/rot
python deploy.py  -c results/coordiff_real/coordiff_mlp_rms_leaf/put_new5_gen_plane_3000

python deploy_silky.py  -c results/coordiff_real/coordiff_mlp_rms_leaf/put_new6_gen_plane

python deploy_simulate.py  -c results/coordiff_real/coordiff_mlp_rms_leaf/put_new6_gen_scale_all
python deploy_simulate.py  -c results/coordiff_real/coordiff_mlp_rms_leaf/teapot_coaster_single_01234
python deploy_hand_simulate.py  -c results/coordiff_real/coordiff_mlp_rms_leaf/hand_put1_gen_scale_12
python deploy_hand.py  -c results/coordiff_real/coordiff_mlp_rms_leaf/hand_put1