import numpy as np
import cv2
import pytorch3d as pt3d
import torch

def normalize_quaternion(q):
    """
    归一化四元数
    :param q: 四元数，格式为 [qx, qy, qz, qw] 或 np.array
    :return: 归一化后的四元数
    """
    q = np.asarray(q)
    norm = np.linalg.norm(q)
    if norm < 1e-10:  # 处理零四元数（无效情况）
        return np.array([0.0, 0.0, 0.0, 1.0])  # 返回默认单位四元数
    return q / norm

def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    q = [qx, qy, qz, qw]
    """
    q = normalize_quaternion(q)
    qx, qy, qz, qw = q

    # assert is_valid_quaternion(q), "Invalid quaternion: norm(q) != 1"


    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    
    return R

def rpy_to_rotation_matrix(roll, pitch, yaw):
    """
    将RPY角（Roll, Pitch, Yaw）转换为3x3旋转矩阵。

    参数:
        roll  - 绕X轴的旋转角（弧度）
        pitch - 绕Y轴的旋转角（弧度）
        yaw   - 绕Z轴的旋转角（弧度）

    返回:
        3x3 numpy 旋转矩阵
    """
    # 计算每个角度的正弦和余弦值
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # 旋转矩阵 (ZYX顺序)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,               cp * cr]
    ])
    assert is_valid_rotation_matrix(R), "transfrom: Invalid rotation matrix"

    return R
def T_from_6D(T_6D, rot_type):
    pos = T_6D[:3]  # 提取位置部分

    if rot_type == 'quat':
        rot = quaternion_to_rotation_matrix(T_6D[3:])  # 将四元数转换为旋转矩阵
    elif rot_type == 'rpy':
        rot = rpy_to_rotation_matrix(T_6D[3:])  # 将RPY转换为旋转矩阵
    elif rot_type == '6d':
        rot = pt3d.rotation_6d_to_matrix(torch.from_numpy(T_6D[3:].reshape(1, -1))).numpy().reshape(3, 3)
    elif rot_type == 'rotvec':  # 从旋转向量恢复旋转矩阵
        rot, _ = cv2.Rodrigues(T_6D[3:])  # 从旋转向量转换为旋转矩阵
    else:
        raise NotImplementedError("Unsupported rotation type")
    
    # 构建 4x4 变换矩阵
    T_matrix = np.eye(4)
    T_matrix[:3, 3] = pos  # 设置位置
    T_matrix[:3, :3] = rot  # 设置旋转部分
    
    return T_matrix

T_b_cam_init_6D = [ 1.01672076, 0.01471327, 0.4861624, -1.6358021, -1.7608583, 0.83154876]
T_cam_o_init_6D = [-0.16155991, 0.10301057, 0.60966006, 1.12944049, 0.15459164, 1.67414834]
T_b_o_init_6D  = [ 0.62102, -0.1725326, 0.02058963 -1.80554939, 0.6666671, 1.73845953]
T_o_hand_6D = [-0.08664398, 0.01353264, 0.03777754 , 0.4718366, 1.34144622, 0.7935634 ]
T_hand_g_init_6D = [0., 0. , 0.07,  0. , 0. , 0.  ]

T_cam_o_init = T_from_6D(T_cam_o_init_6D)
T_o_hand = T_from_6D(T_o_hand_6D)
T_hand_g_init = T_from_6D(T_hand_g_init_6D)

T_cam_o_init_inv = np.linalg.inv(T_cam_o_init)
T_o_hand_inv = np.linalg.inv(T_o_hand)
T_hand_g_init_inv = np.linalg.inv(T_hand_g_init)

# 计算 (T_cam_o_init @ T_o_hand @ T_hand_g_init) 的逆
T_combined_inv = T_hand_g_init_inv @ T_o_hand_inv @ T_cam_o_init_inv

T_combined_inv = T_hand_g_init_inv @ T_o_hand_inv @ T_cam_o_init_inv

# 计算 T_b_cam_init
T_b_cam_init = T_b_g_action @ T_combined_inv