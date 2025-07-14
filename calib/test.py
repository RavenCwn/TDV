import urx
import numpy as np
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper

def euler_to_rotation_matrix(rpy):
    """
    将 roll, pitch, yaw（绕 X, Y, Z 轴的旋转角度）转换为旋转矩阵
    """
    roll, pitch, yaw = rpy
    # 绕 X 轴旋转的矩阵
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # 绕 Y 轴旋转的矩阵
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # 绕 Z 轴旋转的矩阵
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 旋转矩阵 = R_z * R_y * R_x
    rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
    return rotation_matrix

def rotation_matrix_to_rotvec(rotation_matrix):
    """
    从旋转矩阵转换为旋转向量
    """
    # 旋转矩阵的迹（对角线元素之和）
    trace = np.trace(rotation_matrix)

    # 旋转角度（弧度）
    theta = np.arccos((trace - 1) / 2)

    # 计算旋转轴（旋转矩阵的反对称部分）
    if np.sin(theta) != 0:
        r11 = rotation_matrix[2, 1] - rotation_matrix[1, 2]
        r12 = rotation_matrix[0, 2] - rotation_matrix[2, 0]
        r13 = rotation_matrix[1, 0] - rotation_matrix[0, 1]

        # 旋转轴的方向
        axis = np.array([r11, r12, r13]) / (2 * np.sin(theta))
    else:
        # 如果角度是 0 或 180，旋转轴可以是任意方向
        axis = np.array([1, 0, 0])  # 任意方向

    # 旋转向量：旋转轴 * 旋转角度
    rotvec = theta * axis
    return rotvec

if __name__ == "__main__":

    rob = urx.Robot("192.168.101.101") # your robot ip
    rob.set_payload(0.5, (0,0,0))
    rob.set_tcp((0, 0, 0.23, 0, 0, 0))   # tool center point
    robotiqgrip = Robotiq_Two_Finger_Gripper(rob, force=0, socket_host="192.168.101.101")
    robotiqgrip.open_gripper()
    robotiqgrip.close_gripper()

    # reset
    joint_angles = np.array([0., -90, -60, -120, 90, 0])
    joint_angles = joint_angles / 180 * 3.14159
    rob.movej(joint_angles, acc=0.2, vel=0.3, wait=True)

    joint_angles = np.array([0., -80, -60, -120, 90, 0])
    joint_angles = joint_angles / 180 * 3.14159
    rob.movej(joint_angles, acc=0.2, vel=0.3, wait=True)



    # xyzrpy = (0.5, 0.2, 0.2, np.pi/2, 0, 0)
    # vec = rotation_matrix_to_rotvec(euler_to_rotation_matrix(xyzrpy[3:]))

    # xyzvec = np.array([*xyzrpy[:3], *vec])

    # robotiqgrip.robot.movel(xyzvec, acc=0.5, vel=0.5)

    robotiqgrip.robot.close()