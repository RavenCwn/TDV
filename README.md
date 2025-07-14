# TbTd: Trajectory-based Transformer Diffusion

该工作主要是完成基于轨迹的diffusion model

## 环境安装
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

## 仿真实验

### 1. 收集数据

```
# 这里一定要用自己的库，相对于官方的做出了部分修改
git clone git@github.com:inFpZero/RLbench_Coordiff.git

bash collected_data.sh
```

### 2. 训练

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

### 3. 验证

```
bash bash_eval.sh
```



## 实物实验

### 1. 跟踪与扩散模型环境配置指南（Strict: Python 3.10 + CUDA 12.1）

#### 🧩 环境要求此处有两个环境注意

- Python: `3.10`
- CUDA: `12.1`（用于跟踪模块）
- CUDA: `11.8`（用于 Diffusion 模块）

#### 📦 跟踪模块环境配置（含 FoundationPose + SAM2）

##### 1️⃣ 安装 Eigen

```bash
conda install conda-forge::eigen=3.4.0
```

⚠️ 若遇到 `fatal error: Eigen/Dense: No such file or directory` 报错：

- 找到 Conda 安装路径下的 eigen（一般为：`<miniconda_path>/envs/samfp/include/eigen3`）

- 修改 `FoundationPose/bundlesdf/mycuda/setup.py` 中的 `include_dirs`，将 `eigen` 路径添加进去

  > 推荐将 eigen 路径 **放在前面**

##### 2️⃣ 安装基础依赖

```bash
python -m pip install -r requirements.txt
pip install torch==2.3.1 torchvision==0.18.1 torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

⚠️ 如遇 `xformers` 报错，取消requirements.txt中的版本限制。

##### 3️⃣ 安装 NVDiffRast

```bash
pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
```

##### 4️⃣ 安装 Kaolin（适用于 PyTorch 2.3.1 + CUDA 12.1）

```bash
pip install --quiet --no-cache-dir kaolin==0.16.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.3.1_cu121.html
```

##### 5️⃣ 安装 PyTorch3D

```bash
pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt231/download.html
```

##### 6️⃣ 编译 FoundationPose 本地模块

```bash
cd Foundationpose
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```

##### 7️⃣ 添加 CUDA 环境变量

```bash
export CUDA_HOME=/usr/local/cuda-12.1/
```

##### 8️⃣ 安装 SAM2

```bash
pip install --no-build-isolation -e .
```

##### 9️⃣ 安装 Grounding DINO

```bash
pip install --no-build-isolation -e grounding_dino
```

##### 🔟 安装缺失依赖包

```bash
pip install opencv-python supervision transformers
```

##### 1️⃣1️⃣ 下载所有模型权重

```bash
cd AutoPoseEstimator_simple/Foundationpose
python download_weights.py


cd Grounded_SAM_2/checkpoints
bash download_ckpts.sh
cd gdino_checkpoints
bash download_ckpts.sh
```

---

## 

### 2. 数据采集

### 2. 数据增强 + 训练 Diffusion 模型

训练真实数据

```
bash all_obj_xxx.sh	
```

### 3. 实物验证

#### 3.1 需开启跟踪模块

##### ▶️ 跟踪模块执行方法

确保已修改 `run_with_camera_two_GD_auto.py` 中：

- 目标物体 & 参考物体名称
- Mesh 路径、Prompt 等

执行命令：

```bash
source cuda_source.sh
python run_with_camera_two_GD_auto.py
```

#### 3.2 推理模块

需要利用 ur5 和 6DOF Pose Estimation模块

```
python deploy_silky_auto.py -c results/coordiff_real/coordiff_mlp_rms_leaf/xxx
python deploy_hand.py  -c results/coordiff_real/coordiff_mlp_rms_leaf/hand_put1
```
