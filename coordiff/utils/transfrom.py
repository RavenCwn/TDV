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

def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    q = [qx, qy, qz, qw]
    """
    qx, qy, qz, qw = q
    assert is_valid_quaternion(q), "Invalid quaternion: norm(q) != 1"

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
    assert is_valid_rotation_matrix(R), "transfrom: Invalid rotation matrix"
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
    
    return np.array([qx, qy, qz, qw])

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


def compute_relative_pose_input(move_obj, ref_obj, rot_type='6d'):
   
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