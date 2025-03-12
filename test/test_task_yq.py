import numpy as np
import os
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import CloseJar, PutGroceriesInCupboard
from rlbench.tasks.basketball_in_hoop import BasketballInHoop
from rlbench.tasks.beat_the_buzz import BeatTheBuzz
from rlbench.tasks.block_pyramid import BlockPyramid
from rlbench.tasks.change_channel import ChangeChannel
from rlbench.tasks.change_clock import ChangeClock
from rlbench.tasks.close_box import CloseBox
from rlbench.tasks.close_door import CloseDoor
from rlbench.tasks.close_drawer import CloseDrawer
from rlbench.tasks.close_fridge import CloseFridge
from rlbench.tasks.close_grill import CloseGrill
from rlbench.tasks.close_jar import CloseJar
from rlbench.tasks.close_laptop_lid import CloseLaptopLid
from rlbench.tasks.close_microwave import CloseMicrowave
from rlbench.tasks.empty_container import EmptyContainer
from rlbench.tasks.empty_dishwasher import EmptyDishwasher
from rlbench.tasks.get_ice_from_fridge import GetIceFromFridge
from rlbench.tasks.hang_frame_on_hanger import HangFrameOnHanger
from rlbench.tasks.hit_ball_with_queue import HitBallWithQueue
from rlbench.tasks.hockey import Hockey
from rlbench.tasks.insert_onto_square_peg import InsertOntoSquarePeg
from rlbench.tasks.insert_usb_in_computer import InsertUsbInComputer
from rlbench.tasks.lamp_off import LampOff
from rlbench.tasks.lamp_on import LampOn
from rlbench.tasks.lift_numbered_block import LiftNumberedBlock
from rlbench.tasks.light_bulb_in import LightBulbIn
from rlbench.tasks.light_bulb_out import LightBulbOut
from rlbench.tasks.meat_off_grill import MeatOffGrill
from rlbench.tasks.meat_on_grill import MeatOnGrill
from rlbench.tasks.move_hanger import MoveHanger
from rlbench.tasks.open_box import OpenBox
from rlbench.tasks.open_door import OpenDoor
from rlbench.tasks.open_drawer import OpenDrawer
from rlbench.tasks.open_fridge import OpenFridge
from rlbench.tasks.open_grill import OpenGrill
from rlbench.tasks.open_jar import OpenJar
from rlbench.tasks.open_microwave import OpenMicrowave
from rlbench.tasks.open_oven import OpenOven
from rlbench.tasks.open_washing_machine import OpenWashingMachine
from rlbench.tasks.open_window import OpenWindow
from rlbench.tasks.open_wine_bottle import OpenWineBottle
from rlbench.tasks.phone_on_base import PhoneOnBase
from rlbench.tasks.pick_and_lift import PickAndLift
from rlbench.tasks.pick_and_lift_small import PickAndLiftSmall
from rlbench.tasks.pick_up_cup import PickUpCup
from rlbench.tasks.place_cups import PlaceCups
from rlbench.tasks.place_hanger_on_rack import PlaceHangerOnRack
from rlbench.tasks.place_shape_in_shape_sorter import PlaceShapeInShapeSorter
from rlbench.tasks.play_jenga import PlayJenga
from rlbench.tasks.plug_charger_in_power_supply import PlugChargerInPowerSupply
from rlbench.tasks.pour_from_cup_to_cup import PourFromCupToCup
from rlbench.tasks.press_switch import PressSwitch
from rlbench.tasks.push_button import PushButton
from rlbench.tasks.push_buttons import PushButtons
from rlbench.tasks.put_all_groceries_in_cupboard import \
    PutAllGroceriesInCupboard
from rlbench.tasks.put_books_on_bookshelf import PutBooksOnBookshelf
from rlbench.tasks.put_bottle_in_fridge import PutBottleInFridge
from rlbench.tasks.put_groceries_in_cupboard import PutGroceriesInCupboard
from rlbench.tasks.put_item_in_drawer import PutItemInDrawer
from rlbench.tasks.put_knife_in_knife_block import PutKnifeInKnifeBlock
from rlbench.tasks.put_knife_on_chopping_board import PutKnifeOnChoppingBoard
from rlbench.tasks.put_money_in_safe import PutMoneyInSafe
from rlbench.tasks.put_plate_in_colored_dish_rack import \
    PutPlateInColoredDishRack
from rlbench.tasks.put_rubbish_in_bin import PutRubbishInBin
from rlbench.tasks.put_shoes_in_box import PutShoesInBox
from rlbench.tasks.put_toilet_roll_on_stand import PutToiletRollOnStand
from rlbench.tasks.put_tray_in_oven import PutTrayInOven
from rlbench.tasks.put_umbrella_in_umbrella_stand import \
    PutUmbrellaInUmbrellaStand
from rlbench.tasks.reach_and_drag import ReachAndDrag
from rlbench.tasks.reach_target import ReachTarget
from rlbench.tasks.remove_cups import RemoveCups
from rlbench.tasks.scoop_with_spatula import ScoopWithSpatula
from rlbench.tasks.screw_nail import ScrewNail
from rlbench.tasks.set_the_table import SetTheTable
from rlbench.tasks.setup_checkers import SetupCheckers
from rlbench.tasks.setup_chess import SetupChess
from rlbench.tasks.slide_block_to_target import SlideBlockToTarget
from rlbench.tasks.slide_cabinet_open import SlideCabinetOpen
from rlbench.tasks.slide_cabinet_open_and_place_cups import \
    SlideCabinetOpenAndPlaceCups
from rlbench.tasks.solve_puzzle import SolvePuzzle
from rlbench.tasks.stack_blocks import StackBlocks
from rlbench.tasks.stack_chairs import StackChairs
from rlbench.tasks.stack_cups import StackCups
from rlbench.tasks.stack_wine import StackWine
from rlbench.tasks.straighten_rope import StraightenRope
from rlbench.tasks.sweep_to_dustpan import SweepToDustpan
from rlbench.tasks.take_cup_out_from_cabinet import TakeCupOutFromCabinet
from rlbench.tasks.take_frame_off_hanger import TakeFrameOffHanger
from rlbench.tasks.take_item_out_of_drawer import TakeItemOutOfDrawer
from rlbench.tasks.take_lid_off_saucepan import TakeLidOffSaucepan
from rlbench.tasks.take_money_out_safe import TakeMoneyOutSafe
from rlbench.tasks.take_off_weighing_scales import TakeOffWeighingScales
from rlbench.tasks.take_plate_off_colored_dish_rack import \
    TakePlateOffColoredDishRack
from rlbench.tasks.take_shoes_out_of_box import TakeShoesOutOfBox
from rlbench.tasks.take_toilet_roll_off_stand import TakeToiletRollOffStand
from rlbench.tasks.take_tray_out_of_oven import TakeTrayOutOfOven
from rlbench.tasks.take_umbrella_out_of_umbrella_stand import \
    TakeUmbrellaOutOfUmbrellaStand
from rlbench.tasks.take_usb_out_of_computer import TakeUsbOutOfComputer
from rlbench.tasks.toilet_seat_down import ToiletSeatDown
from rlbench.tasks.toilet_seat_up import ToiletSeatUp
from rlbench.tasks.turn_oven_on import TurnOvenOn
from rlbench.tasks.turn_tap import TurnTap
from rlbench.tasks.tv_on import TvOn
from rlbench.tasks.unplug_charger import UnplugCharger
from rlbench.tasks.water_plants import WaterPlants
from rlbench.tasks.weighing_scales import WeighingScales
from rlbench.tasks.wipe_desk import WipeDesk
import open3d as o3d

import rlbench.tasks as tasks
import inspect
# from rlbench.demo_loader import get_stored_demos
# from rlbench.tasks import TASKS

# from rlbench.tasks import TASKSfor task in TASKS:
#     print(f"{task.__name__}: {inspect.getfile(task)}")

# To use 'saved' demos, set the path below, and set live_demos=False

# import numpy as np
# from pyrep.objects.dummy import Dummy

# # 获取机械臂基座对象（你可以查看 Scene Hierarchy 找到正确的名称）
# robot_base = Dummy('Panda_link0_visual')  # 如果用的是 UR5，可能需要改为 'UR5_base'

# # 获取基座的变换矩阵 (4x4)
# T_base = np.array(robot_base.get_matrix()).reshape(4, 4)

# print("基座的变换矩阵:\n", T_base)
# import ipdb; ipdb.set_trace()
# from pyrep.objects.dummy import Dummy

# # 尝试不同的基座名称
# base_names = ['Panda', 'Panda_link0_visual']  # 可能的基座名称

# for name in base_names:
#     try:
#         robot_base = Dummy(name)
#         print(f"成功找到基座: {name}")
#         T_base = robot_base.get_matrix()
#         print("基座变换矩阵:\n", T_base)
#         break  # 找到正确的基座后跳出循环
#     except:
#         print(f"{name} 不存在")



live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
env = Environment(
    action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
    obs_config=ObservationConfig(),
    headless=False
)

env.launch()

# task = env.get_task(PutGroceriesInCupboard)
task = env.get_task(PourFromCupToCup)
# task.set_variation(6)
descriptions, obs = task.reset()
waypoints = task._task.get_waypoints()
print(waypoints)
import open3d as o3d
def transform_point_cloud(pc, T):
    """ 将点云从相机坐标系变换到基坐标系 """
    pc_h = np.hstack((pc, np.ones((pc.shape[0], 1))))  # 扩展为齐次坐标
    pc_transformed = np.dot(T, pc_h.T).T  # 进行矩阵变换
    return pc_transformed[:, :3] / pc_transformed[:, 3:]  # 归一化

for i, point in enumerate(waypoints):
    point.start_of_path()
    if point.skip:
        continue
    path = point.get_path()
    done = False
    success = False
    gripper_open = 1
    while not done:
        done = path.step()
        task._scene.step()
        task._scene._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
        success, term = task._task.success()
        obs = task._scene.get_observation()

        from pyrep.objects.vision_sensor import VisionSensor

        # 选择一个相机，例如左肩相机
        # cam = VisionSensor('cam_over_shoulder_left')

        # # 获取相机参数
        # resolution = cam.get_resolution()  # (width, height)
        # fov = cam.get_fov()  # 视场角（Field of View，单位：度）
        # near_clip = cam.get_near_clipping_plane()  # 近平面
        # far_clip = cam.get_far_clipping_plane()  # 远平面

        # # 计算焦距 fx, fy（根据 pinhole camera model）
        # width, height = resolution
        # fx = width / (2.0 * np.tan(np.radians(fov) / 2.0))
        # fy = height / (2.0 * np.tan(np.radians(fov) / 2.0))

        # # 计算光心 cx, cy
        # cx = width / 2.0
        # cy = height / 2.0

        # # 打印相机内参
        # print("相机分辨率: ", resolution)
        # print("相机 FOV（视场角）: ", fov)
        # print("相机近平面: ", near_clip)
        # print("相机远平面: ", far_clip)
        # print("内参矩阵 K:")
        # print(f"[[{fx},  0, {cx}],")
        # print(f" [ 0, {fy}, {cy}],")
        # print(" [ 0,  0,  1]]")

        import imageio
        depth_image = obs.right_shoulder_depth
        import numpy as np

        # 假设 depth_image 是 obs.right_shoulder_depth
        np.save("/home/user/YQ/anygrasp_sdk_use415front/right_shoulder_depth.npy", depth_image)

        # right_rgb_uint8 = (right_rgb * 255).astype(np.uint8)
        # 保存为 PNG 文件
        # 重新调整形状为 (height, width, 3)
        # right_rgb_uint8 = right_rgb_uint8.reshape((height, width, 3))
        left_rgb = obs.left_shoulder_rgb.reshape((-1, 3)) / 255.0
        right_rgb = obs.right_shoulder_rgb.reshape((-1, 3)) / 255.0
        right_rgb_uint8 = (right_rgb * 255).astype(np.uint8)

        # 保存为 JPG 或 PNG
        imageio.imwrite("/home/user/YQ/anygrasp_sdk_use415front/right_shoulder_png.jpg", obs.right_shoulder_rgb)  # 保存 JPG
        # 直接访问 scene 以获取相机
        scene = task._scene  # 访问场景对象
        # 列出所有可用的相机
        # 获取右肩相机
        right_shoulder_camera = scene._cam_over_shoulder_right.get_intrinsic_matrix()
        print("im",right_shoulder_camera)
        depth_image = obs.right_shoulder_depth
        print(depth_image.shape)
        #_cams['right_shoulder']
        # 获取相机内参
        intrinsics = right_shoulder_camera.get_intrinsics()
        # 打印内参
        print(f"fx: {intrinsics.fx}, fy: {intrinsics.fy}, cx: {intrinsics.cx}, cy: {intrinsics.cy}")
        # 生成相机内参矩阵 K
        import numpy as np
        K = np.array([[intrinsics.fx, 0, intrinsics.cx],
                    [0, intrinsics.fy, intrinsics.cy],
                    [0, 0, 1]])
        print("Intrinsic Matrix (K):\n", K)
        save_path = "/home/user/YQ/anygrasp_sdk_use415front/right_rgb_cup.png"
        imageio.imwrite(save_path, right_rgb_uint8)
        overhead_rgb = obs.overhead_rgb.reshape((-1, 3)) / 255.0
        wrist_rgb = obs.wrist_rgb.reshape((-1, 3)) / 255.0
        front_rgb = obs.front_rgb.reshape((-1, 3)) / 255.0
        left_rgb = obs.left_rgb.reshape((-1, 3)) / 255.0
        right_rgb = obs.right_rgb.reshape((-1, 3)) / 255.0
        # 按照 hstack 方式拼接齐次坐标（不做变换）
        left_pc = np.hstack((left_pc, np.ones((left_pc.shape[0], 1))))[:, :3]
        right_pc = np.hstack((right_pc, np.ones((right_pc.shape[0], 1))))[:, :3]
        overhead_pc = np.hstack((overhead_pc, np.ones((overhead_pc.shape[0], 1))))[:, :3]
        wrist_pc = np.hstack((wrist_pc, np.ones((wrist_pc.shape[0], 1))))[:, :3]
        front_pc = np.hstack((front_pc, np.ones((front_pc.shape[0], 1))))[:, :3]

        # 创建 Open3D 点云对象
        def create_point_cloud(pc, color):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            pcd.colors = o3d.utility.Vector3dVector(color)
            return pcd

        pcd_left = create_point_cloud(left_pc, left_rgb)
        pcd_right = create_point_cloud(right_pc, right_rgb)
        pcd_overhead = create_point_cloud(overhead_pc, overhead_rgb)
        pcd_wrist = create_point_cloud(wrist_pc, wrist_rgb)
        pcd_front = create_point_cloud(front_pc, front_rgb)
        merged_pcd = pcd_left + pcd_right  + pcd_wrist 
        import numpy as np

        # Extract camera pose (extrinsic matrix) for the left shoulder camera
        left_camera_pose = obs.misc["left_shoulder_camera_extrinsics"]
        right_camera_pose = obs.misc["right_shoulder_camera_extrinsics"]
        overhead_camera_pose = obs.misc["overhead_camera_extrinsics"]
        wrist_camera_pose = obs.misc["wrist_camera_extrinsics"]
        front_camera_pose = obs.misc["front_camera_extrinsics"]

        # Print the extrinsic matrices
        print("Left Shoulder Camera Extrinsics:\n", np.array(left_camera_pose))
        print("Right Shoulder Camera Extrinsics:\n", np.array(right_camera_pose))
        # print("Overhead Camera Extrinsics:\n", np.array(overhead_camera_pose))
        # print("Wrist Camera Extrinsics:\n", np.array(wrist))
        # 创建基坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        save_dir = "/home/user/YQ/anygrasp_sdk_use415front"
        overhead_camera_pose = np.array(obs.misc["overhead_camera_extrinsics"])  # 4x4 matrix
        # Compute the inverse (world to camera transformation)
        world_to_camera = np.linalg.inv(right_camera_pose)
        # Apply the transformation to the point cloud
        transformed_pcd = merged_pcd.transform(world_to_camera)
        # Create a coordinate frame for the overhead camera
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        coordinate_frame.transform(world_to_camera)
        # Visualize the transformed point cloud in the overhead camera frame
        o3d.visualization.draw_geometries([transformed_pcd, coordinate_frame], 
                                        window_name="Point Cloud in Overhead Camera Frame")
        # Save the transformed point cloud
        save_path = "/home/user/YQ/anygrasp_sdk_use415front/transformed_to_overhead.ply"
        o3d.io.write_point_cloud(save_path, transformed_pcd)
        print(f"✅ Transformed point cloud saved at: {save_path}")
        left_pc = obs.left_shoulder_point_cloud
        left_pc = left_pc.reshape((-1, 3))
        # # save rgb and point cloud
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(left_pc)
        # open3d.visualization.draw_geometries([pcd])
        import ipdb; ipdb.set_trace()
    point.end_of_path()
    path.clear_visualization()
env.shutdown()
# yong rgb front nage camera