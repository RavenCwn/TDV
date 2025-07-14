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
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
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

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
    
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
        mask = ids == 582
        corners = [corners[i] for i in range(len(ids)) if mask[i]]
        ids = [ids[i] for i in range(len(ids)) if mask[i]]

        if len(corners) > 0:
            # 确保 ids 是 numpy 数组
            ids = np.array(ids)
            tag_size = 0.048
            # 估计出aruco码的位姿，0.17是marker的边长，单位是米
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, tag_size, intr_matrix, intr_coeffs)
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
    rob = urx.Robot("192.168.101.101") # your robot ip
    rob.set_payload(0.5, (0,0,0))
    # rob.set_tcp((0,0,0.19,0, 0,0))   # tool center point

    tool_ini_joints= np.deg2rad(np.array([0., -90, -60, -120, 90, 0]))
    l = 0.05                    # unit: meter
    v = 0.08                    # default: 0.04
    a = 0.1
    aj= 0.8
    vj= 0.5
    # rob.movej(tool_ini_joints, aj, vj)
    print("Robot moved to initial position.",rob.getl())
    

    try:

        while True:
            rgb, depth, intr_matrix, intr_coeffs = get_aligned_images()
            print(intr_coeffs)

            intr_matrix = np.array([[color_intrinsics.fx,        0,             color_intrinsics.ppx],
                                    [ 0 ,              color_intrinsics.fy, color_intrinsics.ppy],
                                    [ 0 ,          0  , 1]])
            # intr_coeffs = np.array([[1.64081007e-01, -4.91975188e-01, -1.13262527e-03,  5.55912302e-05,4.43834335e-01]])
            print("intr_matrix:\n")
            print(intr_matrix)
            print("intr_coeffs:\n")
            print(intr_coeffs)

            marker = get_realsense_marker(rgb,intr_matrix, intr_coeffs)
            key = cv2.waitKey(1)


            if key & 0xFF == ord('q') or key == 27:
                pipeline.stop()
                break
            # 按键盘r记录g-b，m-c位姿
            elif key == ord('r'):  # record
                gripper_poses.append(get_jaka_gripper()) # 返回的显示t后世p
                # assert marker is not None, "dataset is useless"
                if marker is not None:

                    marker_poses.append(marker)
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
                    R_gripper_in_base.append(gripper_rot) #欧拉角到旋转矩阵；# 表示为按照xyz
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

                T_c_b = np.eye(4)
                T_c_b[:3, :3] = R_c_b
                T_c_b[:3, 3] = t_c_b.flatten()


                print('T_c_b\n', T_c_b)
                break
        cv2.destroyAllWindows()
    finally:
        rob.close()

# cam2 to base
""" [[ 0.73649722 -0.40043978  0.54517871 -0.91937018]
 [-0.67190532 -0.52624003  0.52116664 -0.87165106]
 [ 0.078199   -0.75014626 -0.65663194  0.44615543]
 [ 0.          0.          0.          0.        ]]"""

"""T_c_b
 [[ 0.71270673 -0.44896236  0.53896375 -0.90638918]
 [-0.70044156 -0.41406409  0.58131966 -0.85957366]
 [-0.03782511 -0.79182304 -0.60957816  0.40518278]
 [ 0.          0.          0.          0.        ]]
 """

"""
T_c_b
 [[ 0.66894062 -0.54275079  0.50787797 -0.93745781]
 [-0.7399085  -0.42085807  0.52479891 -0.86632757]
 [-0.07109048 -0.72684253 -0.68311498  0.42707418]
 [ 0.          0.          0.          1.        ]]
"""

"""
T_c_b
 [[ 0.05079699  0.68025314 -0.73121497  1.1165818 ]
 [ 0.99840218 -0.05273617  0.02029756 -0.02049759]
 [-0.024754   -0.73107767 -0.68184505  0.48054556]
 [ 0.          0.          0.          1.        ]]
"""
"""
4-1
 T_c_b
  [[-0.21224676  0.47384972 -0.85464481  1.12760912]
 [ 0.94851396  0.31030168 -0.06351486  0.08316461]
 [ 0.23510122 -0.82412336 -0.5153136   0.29864352]
 [ 0.          0.          0.          1.        ]]


 [[-0.03778028  0.72090825 -0.69199996  1.15121582]
 [ 0.99730288 -0.01640515 -0.07153901  0.03459794]
 [-0.06292542 -0.69283632 -0.71834409  0.44752758]
 [ 0.          0.          0.          1.        ]]

  [[ 0.02595616  0.7553074  -0.65485648  1.15195756]
 [ 0.99553264 -0.07901854 -0.05168015  0.0032606 ]
 [-0.0907802  -0.65058958 -0.75398419  0.47306478]
 [ 0.          0.          0.          1.        ]]
 T_c_b
 [[ 0.06893862  0.74722182 -0.66098942  1.14820299]
 [ 0.99419417 -0.10632706 -0.01650786  0.03131992]
 [-0.0826161  -0.6560138  -0.75021362  0.47491357]
 [ 0.          0.          0.          1.        ]]
 4.2
  [[-0.21730758  0.28750481 -0.93280137  1.16424812]
 [ 0.96633846  0.19820086 -0.16403168  0.23785056]
 [ 0.13772214 -0.93704717 -0.3208975   0.34978075]
 [ 0.          0.          0.          1.        ]]

  [[ 0.05918451  0.70215215 -0.70956293  1.16206772]
 [ 0.99748821 -0.06930829  0.01461588 -0.11324448]
 [-0.03891602 -0.70864569 -0.70449047  0.46019043]
 [ 0.          0.          0.          1.        ]]

  [[ 0.07904223  0.60178306 -0.79473862  1.18658202]
 [ 0.99682281 -0.03985282  0.06896399 -0.07599432]
 [ 0.00982879 -0.79766465 -0.60302114  0.45570748]
 [ 0.          0.          0.          1.        ]]

  [[ 0.15828932  0.62753774 -0.76232596  1.10553288]
 [ 0.98695402 -0.1235697   0.10320996 -0.16086179]
 [-0.02943225 -0.7687177  -0.63891066  0.35897454]
 [ 0.          0.          0.          1.        ]]

  [[ 0.44071484 -0.35324442 -0.82522046  0.89314913]
 [ 0.89588782  0.11556216  0.42898764 -0.21099543]
 [-0.05617324 -0.92836618  0.36739733 -0.61356391]
 [ 0.          0.          0.          1.        ]]

  [[-0.13564447  0.54771215 -0.82559795  1.05006415]
 [ 0.98848963  0.13116635 -0.0753899   0.01353169]
 [ 0.06699871 -0.82632124 -0.55919977  0.32844856]
 [ 0.          0.          0.          1.        ]]

4-9_20:24
  [[ 0.00824559  0.55120538 -0.83432886  1.02815004]
 [ 0.99795435 -0.05743326 -0.02808102 -0.00693304]
 [-0.06339664 -0.83239056 -0.55055137  0.39449762]
 [ 0.          0.          0.          1.        ]]
"""


