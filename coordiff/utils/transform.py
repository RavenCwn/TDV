import numpy as np
import torch
import pytorch3d.transforms as pt3d

def is_valid_quaternion(q):
    # 解包四元数分量
    qx, qy, qz, qw = q
    # 计算模长平方
    norm_squared = qx**2 + qy**2 + qz**2 + qw**2
    # 允许的浮点误差范围
    epsilon = 1e-6
    # 检查是否接近单位四元数
    return abs(norm_squared - 1.0) < epsilon

def is_valid_rotation_matrix(R):
    # 允许的浮点误差范围
    epsilon = 1e-6
    
    # 检查是否为 3x3 矩阵
    if len(R) != 3 or any(len(row) != 3 for row in R):
        return False
    
    # 计算 R 的转置
    Rt = list(zip(*R))
    
    # 检查 R^T * R 是否接近单位矩阵
    for i in range(3):
        for j in range(3):
            dot_product = sum(R[i][k] * Rt[k][j] for k in range(3))
            # 对角线元素应接近 1，非对角线应接近 0
            if i == j and abs(dot_product - 1.0) > epsilon:
                return False
            elif i != j and abs(dot_product) > epsilon:
                return False
    
    # 检查行列式是否接近 1
    det = (
        R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
        - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
        + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0])
    )
    return abs(det - 1.0) < epsilon

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

def rotation_matrix_to_quaternion(R):
    """
    将旋转矩阵转换为四元数
    返回格式: q = [qx, qy, qz, qw]
    """
    # 处理批量输入
    if R.ndim == 3:  # 检查是否为批量输入
        assert R.shape[1:] == (3, 3), "Input must be of shape (N, 3, 3)"
        
        tr = np.trace(R, axis1=1, axis2=2)  # 计算每个旋转矩阵的迹
        qw = np.zeros(R.shape[0])
        qx = np.zeros(R.shape[0])
        qy = np.zeros(R.shape[0])
        qz = np.zeros(R.shape[0])
        
        # 计算 S
        S = np.sqrt(np.maximum(tr + 1.0, 0)) * 2
        
        # 条件判断
        mask = tr > 0
        qw[mask] = 0.25 * S[mask]
        qx[mask] = (R[mask, 2, 1] - R[mask, 1, 2]) / S[mask]
        qy[mask] = (R[mask, 0, 2] - R[mask, 2, 0]) / S[mask]
        qz[mask] = (R[mask, 1, 0] - R[mask, 0, 1]) / S[mask]

        mask1 = ~mask & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        S1 = np.sqrt(1.0 + R[mask1, 0, 0] - R[mask1, 1, 1] - R[mask1, 2, 2]) * 2
        qw[mask1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / S1
        qx[mask1] = 0.25 * S1
        qy[mask1] = (R[mask1, 0, 1] + R[mask1, 1, 0]) / S1
        qz[mask1] = (R[mask1, 0, 2] + R[mask1, 2, 0]) / S1

        mask2 = ~mask & (R[:, 1, 1] > R[:, 2, 2])
        S2 = np.sqrt(1.0 + R[mask2, 1, 1] - R[mask2, 0, 0] - R[mask2, 2, 2]) * 2
        qw[mask2] = (R[mask2, 0, 2] - R[mask2, 2, 0]) / S2
        qx[mask2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / S2
        qy[mask2] = 0.25 * S2
        qz[mask2] = (R[mask2, 1, 2] + R[mask2, 2, 1]) / S2

        mask3 = ~mask & ~mask1 & ~mask2
        S3 = np.sqrt(1.0 + R[mask3, 2, 2] - R[mask3, 0, 0] - R[mask3, 1, 1]) * 2
        qw[mask3] = (R[mask3, 1, 0] - R[mask3, 0, 1]) / S3
        qx[mask3] = (R[mask3, 0, 2] + R[mask3, 2, 0]) / S3
        qy[mask3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / S3
        qz[mask3] = 0.25 * S3

        q = np.array([qx, qy, qz, qw]).T
        return normalize_quaternion(q)

    else:
        return rotation_matrix_to_quaternion_single(R)  # 处理单个旋转矩阵

def rotation_matrix_to_quaternion_single(R):
    tr = np.trace(R)
    
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
    
    q = np.array([qx, qy, qz, qw])
    return normalize_quaternion(q)


def rotation_matrix_to_rpy(R):
    """
    Convert a 3x3 rotation matrix to roll, pitch, yaw (RPY) angles.
    
    Parameters:
        R (numpy.ndarray): 3x3 rotation matrix.
        
    Returns:
        tuple: (roll, pitch, yaw) in radians.
    """
    assert is_valid_rotation_matrix(R), "transfrom: Invalid rotation matrix"
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    
    return roll, pitch, yaw

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


def compute_relative_pose_T(move_obj, ref_obj):
    T_m, T_r = np.eye(4), np.eye(4)
    pos_m, quat_m = move_obj[:3], move_obj[3:]  # [x,y,z], [qx,qy,qz,qw]
    pos_r, quat_r = ref_obj[:3], ref_obj[3:]
    
    # 四元数转旋转矩阵
    R_m = quaternion_to_rotation_matrix(quat_m)
    R_r = quaternion_to_rotation_matrix(quat_r)
    
    # 构建齐次变换矩阵
    T_m[:3, :3] = R_m
    T_m[:3, 3] = pos_m
    
    T_r[:3, :3] = R_r
    T_r[:3, 3] = pos_r
    
    
    T_r_m = np.linalg.inv(T_r) @ T_m

    return T_r_m


def compute_relative_pose_input(move_obj, ref_obj, rot_type):
   
    T_r_m = compute_relative_pose_T(move_obj, ref_obj)

    pos = T_r_m[:3, 3]

    if rot_type == 'quat':
        rot = rotation_matrix_to_quaternion(T_r_m[:3, :3])
    elif rot_type == 'rpy':
        rot = rotation_matrix_to_rpy(T_r_m[:3, :3])
    elif rot_type == '6d':
        rot = pt3d.matrix_to_rotation_6d(
            torch.from_numpy(T_r_m[:3, :3]).unsqueeze(0)
        ).numpy().flatten()
    else:
        raise NotImplementedError("Unsupported rotation type")

    return np.concatenate([pos, rot])

def compute_relative_traj_input(moving_traj, ref_traj, rot_type='6d'):
    #  (X,Y,Z,Qx,Qy,Qz,Qw)

    relevant_traj = []
    for m, r in zip(moving_traj, ref_traj):
        relative_pose = compute_relative_pose_input(m, r, rot_type)

        relevant_traj.append(relative_pose)
    
    return np.array(relevant_traj)

def compute_relative_traj_smooth_savgol(moving_traj, ref_traj, rot_type='6d', window_length=5, polyorder=2):
    #  (X,Y,Z,Qx,Qy,Qz,Qw)

    relevant_traj = []
    for m, r in zip(moving_traj, ref_traj):
        T_r_m = compute_relative_pose_T(m, r)
        relevant_traj.append(T_r_m)

    # 转换为numpy数组以便进行平滑处理
    relevant_traj = np.array(relevant_traj)  # 形状为 (N, 4, 4)

    # 提取位置和旋转
    positions = relevant_traj[:, :3, 3]
    rotations = np.array([rotation_matrix_to_quaternion(T[:3, :3]) for T in relevant_traj])

    # 使用Savitzky-Golay滤波器进行平滑
    smoothed_positions = savgol_filter(positions, window_length=window_length, polyorder=polyorder, axis=0)
    smoothed_rotations = savgol_filter(rotations, window_length=window_length, polyorder=polyorder, axis=0)

    # 组合位置和旋转
    final_traj = []
    for pos, rot in zip(smoothed_positions, smoothed_rotations):
        final_traj.append(np.concatenate([pos, rot]))

    return np.array(final_traj)

def compute_relative_traj_smooth(moving_traj, ref_traj, rot_type='6d', smoothing_window=3):
    #  (X,Y,Z,Qx,Qy,Qz,Qw)

    relevant_traj = []
    for m, r in zip(moving_traj, ref_traj):
        T_r_m = compute_relative_pose_T(m, r)
        relevant_traj.append(T_r_m)

    # 转换为numpy数组以便进行平滑处理
    relevant_traj = np.array(relevant_traj)  # 形状为 (N, 4, 4)

    # 应用平滑处理（移动平均）
    smoothed_traj = np.copy(relevant_traj)
    for i in range(relevant_traj.shape[0]):
        start = max(0, i - smoothing_window // 2)
        end = min(relevant_traj.shape[0], i + smoothing_window // 2 + 1)
        smoothed_traj[i] = np.mean(relevant_traj[start:end], axis=0)

    # 将平滑后的结果转换为所需的旋转格式
    final_traj = []
    for T in smoothed_traj:
        if rot_type == 'quat':
            rot = rotation_matrix_to_quaternion(T[:3, :3])
        elif rot_type == 'rpy':
            rot = rotation_matrix_to_rpy(T[:3, :3])
        elif rot_type == '6d':
            rot = pt3d.matrix_to_rotation_6d(
                torch.from_numpy(T[:3, :3]).unsqueeze(0)
            ).numpy().flatten()
        else:
            raise NotImplementedError("Unsupported rotation type")
        
        final_traj.append(np.concatenate([T[:3, 3], rot]))  # 组合位置和旋转

    return np.array(final_traj)

def get_abs_action(relative_pose, ref_pose, rot_type='6d', T_o_g=None):
    T_ref = np.eye(4)
    R_ref = quaternion_to_rotation_matrix(ref_pose[3:])
    T_ref[:3, :3] = R_ref
    T_ref[:3, 3] = ref_pose[:3]

    T_relative = np.eye(4)
    T_relative[:3, 3] = relative_pose[:3]
    if rot_type == 'quat':
        rot = quaternion_to_rotation_matrix(relative_pose[3:])
    elif rot_type == 'rpy':
        rot = rpy_to_rotation_matrix(relative_pose[3:])
    elif rot_type == '6d':
        rot = pt3d.rotation_6d_to_matrix(
            torch.from_numpy(relative_pose[3:]).unsqueeze(0)
        ).numpy()
    else:
        raise NotImplementedError("Unsupported rotation type")
    T_relative[:3, :3] = rot

    # get abs action
    T_abs = T_ref @ T_relative

    if T_o_g is not None:
        T_abs = T_abs @ T_o_g 

    pos = T_abs[:3, 3]

    rot = rotation_matrix_to_quaternion(T_abs[:3, :3])

    return np.concatenate([pos, rot])


def get_abs_pose(name, obs, task):
    pose = None
    if "gripper_pose" in name:
        pose = obs.gripper_pose
    else:
        obj = get_obj_by_name(name, task)
        pose = obj.get_pose()
        # obj = Object.get_object(name)
        # pose = obj.get_pose()
    assert pose is not None, f"Can't find the object [{name}] pose in task [{task.name}]"

    if np.isnan(pose).any():
        import ipdb; ipdb.set_trace()

    return pose



def get_obj_by_name(target_obj_name, task):
    # objs = task._base_object.get_objects_in_tree()
    objs_type = task._initial_objs_in_scene
    target_obj = None
    for obj, _ in objs_type:
        if obj.get_name() == target_obj_name:
            target_obj = obj
            break

    return target_obj