import numpy as np
from T import *
# 定义旋转矩阵 R1 和 R2
# R1 为 base_hand的第一帧数据

# T1 为 cam_o
T1 = np.load('/home/user/pgp/CoorDiff/pose/obj_pose.npy')

# T2 为 cam_hand
task_name = 'final_mug'
T2 = np.load(f'/home/user/ljc/WiLoR/datasets/{task_name}/episode_0/hand_poses.npy')
T2 = T2[1]      # 第一帧数据有bug

def inverse_homogeneous(T):
    """
    计算齐次变换矩阵 T 的逆矩阵。
    T 为 4×4 矩阵，格式为 [R, t; 0, 1]
    返回 T^{-1} = [R^T, -R^T t; 0, 1]
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -np.dot(R.T, t)
    return T_inv

# 计算 T1 的逆矩阵
T1_inv = inverse_homogeneous(T1)

# 计算相对变换矩阵 T_rel = T1^{-1} * T2
T_rel = np.dot(T1_inv, T2)

# 按要求格式化输出最终结果
print("计算得到的相对变换矩阵 T1 @ T_rel = T2:")
print("T_rel = np.array([")
for i in range(4):
    row_str = "    [" + ", ".join(f"{T_rel[i, j]:.8f}" for j in range(4)) + "]"
    if i < 3:
        row_str += ","
    print(row_str)
print("])")

# T1为base_hand T2为base_gripper T_rel为hand_gripper
print(f"验证结果T1 @ T_rel = {T1 @ T_rel}")
