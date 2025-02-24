# import jkrc
import math
import time
import pyrealsense2 as rs
import numpy as np
print("ur5_move.py",np.__version__)
import cv2
import urx
import sys
path_to_remove = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)
# 提示没有aruco的看问题汇总
import cv2.aruco as aruco
import transforms3d as tfs
from packaging import version  # Installed with setuptools, so should already be installed in your env.
import logging
import cv2
import numpy as np
from cv2 import aruco


# robot = urx.Robot("192.168.1.101") # your robot ip
#JAKA初始化
# robot = jkrc.RC("192.168.0.104")#机械臂ip地址
#登录
# robot.login()

# realsense初始化,配置摄像头与开启pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 获取深度相机的内参
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
depth_intrinsics = depth_stream.get_intrinsics()

# 获取RGB相机的内参
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
color_intrinsics = color_stream.get_intrinsics()

# 打印RGB相机内参
print("\nRGB相机内参：")
print("焦距 (fx, fy):", color_intrinsics.fx, color_intrinsics.fy)
print("主点 (cx, cy):", color_intrinsics.ppx, color_intrinsics.ppy)
# print("畸变系数:", color_intrinsics.coefficients)

align_to = rs.stream.color
align = rs.align(align_to)
print("align start",align)
#获取jaka的末端位姿，xyz；弧度制rxryrz
def get_jaka_gripper():
    pose = rob.get_pose() 
    print(pose.pos)
    print(pose.orient)
    pos_trans = np.array([pose.pos.x, pose.pos.y, pose.pos.z])
    pos_R = pose.orient
    return  [pos_trans, pos_R]
# 获取对齐的rgb和深度图
def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    # 获取intelrealsense参数
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    # 内参矩阵，转ndarray方便后续opencv直接使用
    intr_matrix = np.array([
        [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
    ])
    # 深度图-16位
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    # 深度图-8位
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
    pos = np.where(depth_image_8bit == 0)
    depth_image_8bit[pos] = 255
    # rgb图
    color_image = np.asanyarray(color_frame.get_data())
    # return: rgb图，深度图，相机内参，相机畸变系数(intr.coeffs)
    return color_image, depth_image, intr_matrix, np.array(intr.coeffs)

def rpy_to_rotation_matrix(roll, pitch, yaw):
    # Roll, Pitch, Yaw 转换为旋转矩阵
    # Roll (x), Pitch (y), Yaw (z)
    
    # 计算每个旋转矩阵
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # 旋转矩阵组合，顺序为绕z, y, x轴旋转
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    return R

def get_realsense_marker(rgb, intr_matrix, intr_coeffs):
    # 获取dictionary，指示位50个
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    
    # 创建detector parameters
    parameters = aruco.DetectorParameters()
    
    # 检测 ArUco 标签
    corners, ids, rejected_img_points = aruco.detectMarkers(rgb, aruco_dict)

    # 输出检测到的Marker ID
    print("marker_ids***************************************", ids)
    print("corners***************************************", corners)
    # 检查是否检测到标记
    # print("corners", corners, len(corners), len(ids), ids)
    if len(corners) > 0:
        # 过滤出 ID 为 582 的标记
        # mask = ids == 582
        # corners = [corners[i] for i in range(len(ids)) if mask[i]]
        # ids = [ids[i] for i in range(len(ids)) if mask[i]]

        if len(corners) > 0:
            # 确保 ids 是 numpy 数组
            ids = np.array(ids)

            # 估计出aruco码的位姿，0.17是marker的边长，单位是米
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.0525, intr_matrix, intr_coeffs)
            print("Rotation Vectors:", rvec, rvec.shape)
            print("Translation Vectors:", tvec, tvec.shape)
            print("markerPoints", markerPoints)
            print("ids",ids)
            # 可视化结果：绘制检测到的标记
            rgb = aruco.drawDetectedMarkers(rgb, corners, ids)

            for i in range(len(ids)):
                # 使用cv2.drawFrameAxes绘制坐标轴，0.1为坐标轴的长度
                rgb = cv2.drawFrameAxes(rgb, intr_matrix, intr_coeffs, rvec[i], tvec[i], 0.1)

            # 使用cv2.imshow显示图像并等待按键事件
            cv2.imshow("Detected ArUco markers", rgb)
            
            # 返回平移和旋转向量（转化为一维数组）
            print("Rotation Vectors:", rvec, ".shape()")
            print("Translation Vectors:", tvec, ".shape()")
            return list(np.reshape(tvec, 3)) + list(np.reshape(rvec, 3))
        else:
            print("Marker with ID 582 not detected!")
            return None
    else:
        print("No markers detected!")
        # 如果没有检测到标记，仍然显示原图
        # cv2.imshow("No Markers Detected", rgb)
        # cv2.waitKey(1)
        return None

def solve_hand_eye(robot_rotations, robot_translations, camera_rotations, camera_translations):
    # 使用OpenCV的calibrateHandEye函数求解手眼标定
    R_eye, t_eye = cv2.calibrateHandEye(
        robot_rotations, robot_translations,
        camera_rotations, camera_translations,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    return R_eye, t_eye


if __name__ == "__main__":
    gripper_poses =[]
    marker_poses = []
    logging.basicConfig(level=logging.WARN) 
    rob = urx.Robot("192.168.1.101") # your robot ip
    rob.set_payload(0.5, (0,0,0))
    rob.set_tcp((0,0,0.19,0, 0,0))   # tool center point

    tool_ini_joints= [0.5428823828697205, -1.9081628958331507, 1.6074042320251465, -1.2644203344928187, -1.6236584822284144, 1.3247456550598145]
    l = 0.05                    # unit: meter
    v = 0.08                    # default: 0.04
    a = 0.1
    aj= 0.8
    vj= 0.5
    rob.movej(tool_ini_joints, aj, vj)
    print("Robot moved to initial position.",rob.getl())


    try:

        while True:
            rgb, depth, intr_matrix, intr_coeffs = get_aligned_images()
            intr_matrix = np.array([[color_intrinsics.fx,        0,             color_intrinsics.ppx],
                                    [ 0 ,              color_intrinsics.fy, color_intrinsics.ppy],
                                    [ 0 ,          0  , 1]])
            intr_coeffs = np.array([[0., 0., 0., 0., 0.]])
            marker = get_realsense_marker(rgb,intr_matrix, intr_coeffs)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                pipeline.stop()
                break
            # 按键盘r记录g-b，m-c位姿
            elif key == ord('r'):
                gripper_poses.append(get_jaka_gripper()) # 返回的显示t后世p
                # assert marker is not None, "dataset is useless"
                if marker is not None:

                    marker_poses.append(marker)

            elif key == ord('w'):  # 向前移动
                # rob.translate((l, 0, 0), a, v) 
                rob.movel((l, 0, 0, 0, 0, 0), a, v, relative=True)
            elif key == ord('s'):  # 向后移动
                rob.movel((-l, 0, 0, 0, 0, 0), a, v, relative=True)
            elif key == ord('a'):  # 向左移动
                rob.movel((0, -l, 0, 0, 0, 0), a, v, relative=True)
            elif key == ord('d'):  # 向右移动
                rob.movel((0, l, 0, 0, 0, 0), a, v, relative=True)
            elif key == ord('z'):  # 向上移动
                rob.movel((0, 0, l, 0, 0, 0), a, v, relative=True)
            elif key == ord('x'):  # 向下移动
                rob.movel((0, 0, -l, 0, 0, 0), a, v, relative=True)
            elif key == ord('i'):  # 俯仰向上
                rob.movel((0, 0, 0, 0, l, 0), a, v, relative=True)
            elif key == ord('k'):  # 俯仰向下
                rob.movel((0, 0, 0, 0, -l, 0), a, v, relative=True)
            elif key == ord('j'):  # 偏航向左
                rob.movel((0, 0, 0, -l, 0, 0), a, v, relative=True)
            elif key == ord('l'):  # 偏航向右
                rob.movel((0, 0, 0, l, 0, 0), a, v, relative=True)
            elif key == ord('u'):  # 滚转向左
                rob.movel((0, 0, 0, 0, 0, -l), a, v, relative=True)
            elif key == ord('o'):  # 滚转向右
                rob.movel((0, 0, 0, 0, 0, l), a, v, relative=True)

            elif key ==ord('c'):
                R_gripper_in_base, R_target_in_camera = [], []
                t_gripper_in_base, t_target_in_camera = [], []
                for marker in marker_poses:
                    #m-c的旋转矩阵和位移矩阵
                    camera_rot = marker[3:6] # 3:6 是旋转向量
                    camera_mat,_ = cv2.Rodrigues((camera_rot[0],camera_rot[1],camera_rot[2])) #旋转矢量到旋转矩阵
                    R_target_in_camera.append(camera_mat)
                    t_target_in_camera.append(np.array(marker[0:3])) 
                for gripper in gripper_poses:

                    # g-b的旋转矩阵和位移矩阵
                    gripper_rot = gripper[1].get_array()
                    gripper_pos = gripper[0]
                    R_gripper_in_base.append(gripper_rot)#欧拉角到旋转矩阵；# 表示为按照xyz
                    t_gripper_in_base.append(gripper_pos) 
                
                print("R_target_in_camera:",R_target_in_camera)
                print("t_target_in_camera:",t_target_in_camera)
                print("R_gripper_in_base:",R_gripper_in_base)
                print("t_gripper_in_base:",t_gripper_in_base)#这个是错的
                # 将矩阵存储到字典中
                data = {
                    'R_target_in_camera': R_target_in_camera,
                    't_target_in_camera': t_target_in_camera,
                    'R_gripper_in_base':   R_gripper_in_base,
                    't_gripper_in_base':   t_gripper_in_base
                }

                # 保存到 .npy 文件
                np.save('calibration_data_without_tcp.npy', data)


                R_gripper_in_base = np.array(R_gripper_in_base)
                t_gripper_in_base = np.array(t_gripper_in_base)
                R_target_in_camera = np.array(R_target_in_camera)
                t_target_in_camera = np.array(t_target_in_camera)

                T_ee_b = np.zeros((len(R_gripper_in_base), 4, 4))
                for i in range(len(T_ee_b)):
                    T_ee_b[i][:3, :3] = R_gripper_in_base[i]
                    T_ee_b[i][:3, 3] = t_gripper_in_base[i]


                T_b_ee = np.zeros_like(T_ee_b)
                for i in range(len(T_ee_b)):
                    T_b_ee[i][:3, :3] = np.transpose(T_ee_b[i][:3, :3])
                    T_b_ee[i][:3, 3] = -np.dot(np.transpose(T_ee_b[i][:3, :3]), T_ee_b[i][:3, 3])

                r_b_ee = T_b_ee[:, :3, :3].reshape(-1, 3, 3)
                t_b_ee = T_b_ee[:, :3, 3].reshape(-1, 3)
                r_t_c = R_target_in_camera.reshape(-1, 3, 3)
                t_t_c = t_target_in_camera.reshape(-1, 3)

                print(len(r_b_ee),len(t_b_ee))
                R_c_b, t_c_b = solve_hand_eye(r_b_ee, t_b_ee, r_t_c, t_t_c)

                print(R_c_b, t_c_b)

                T_c_b = np.zeros((4, 4))
                T_c_b[:3, :3] = R_c_b
                T_c_b[:3, 3] = t_c_b.flatten()


                print('T_c_b\n', T_c_b)
                break
        cv2.destroyAllWindows()
    finally:
        rob.close()
"""
返回的是
T_c_b
 [[ 0.69744345 -0.70970695  0.09944184 -0.31799075]
 [-0.68552771 -0.62026855  0.38120687 -0.67699799]
 [-0.20886452 -0.33404037 -0.91912602  0.85876261]
 [ 0.          0.          0.          0.        ]]
"""