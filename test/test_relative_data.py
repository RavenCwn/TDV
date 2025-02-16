import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import CloseJar

from coordiff.utils.transfrom import *

action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
env = Environment(
    action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
    obs_config=ObservationConfig(),
    headless=False
)

env.launch()

task = env.get_task(CloseJar)


related_0 = np.load("/home/a4090/lfwh/trajectory_diff/coordiff/data3_re/close_jar/train/episode_0022/0/relevant_traj.npy")
related_1 = np.load("/home/a4090/lfwh/trajectory_diff/coordiff/data3_re/close_jar/train/episode_0022/1/relevant_traj.npy")

jar_lid0_pose = np.loadtxt("/home/a4090/lfwh/trajectory_diff/coordiff/mydata3_re/close_jar/episode_0022/0/trajectory/task_shape/jar_lid0.txt", delimiter=' ')[0]
jar0_pose = np.loadtxt("/home/a4090/lfwh/trajectory_diff/coordiff/mydata3_re/close_jar/episode_0022/0/trajectory/task_shape/jar0.txt", delimiter=' ')[0]

print(related_0.shape, related_1.shape)
print(jar_lid0_pose.shape, jar0_pose.shape)


descriptions, obs = task.reset()

for i, relative_pose in enumerate(related_0):
    # print(a, a.shape)
    a = get_abs_action(relative_pose, jar_lid0_pose, rot_type='6d')
    a = np.concatenate([a, np.array([1.])])
    obs, reward, terminate = task.step(a)

gripper_pose = obs.gripper_pose
T_o_g = compute_relative_pose_T(gripper_pose, jar_lid0_pose)

for i, relative_pose in enumerate(related_1):
    # print(a, a.shape)
    a = get_abs_action(relative_pose, jar0_pose, rot_type='6d', T_o_g=T_o_g)
    a = np.concatenate([a, np.array([1.])])
    obs, reward, terminate = task.step(a)


print('Done')
env.shutdown()
