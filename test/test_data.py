import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import CloseJar


# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
env = Environment(
    action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
    obs_config=ObservationConfig(),
    headless=False
)

env.launch()

task = env.get_task(CloseJar)

action_0 = np.loadtxt("/home/a4090/lfwh/trajectory_diff/coordiff/new_data/close_jar/episode0001/data/0/trajectory/gripper_pose.txt", delimiter=' ')
action_1 = np.loadtxt("/home/a4090/lfwh/trajectory_diff/coordiff/new_data/close_jar/episode0001/data/1/trajectory/gripper_pose.txt", delimiter=' ')
# action = np.concatenate([action_0, action_1], axis=0)

action_0 = np.loadtxt("/home/a4090/lfwh/trajectory_diff/coordiff/mydata3_re/close_jar/episode_0000/0/trajectory/gripper_pose.txt", delimiter=' ')
action_1 = np.loadtxt("/home/a4090/lfwh/trajectory_diff/coordiff/mydata3_re/close_jar/episode_0000/1/trajectory/gripper_pose.txt", delimiter=' ')

action = action_0

obs = None
descriptions, obs = task.reset()
waypoints = task._task.get_waypoints()
print(waypoints)

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

    point.end_of_path()

    path.clear_visualization()
    import ipdb; ipdb.set_trace()

    # if len(ext) > 0:
    #     contains_param = False
    #     start_of_bracket = -1
    #     gripper = self.robot.gripper
    #     if 'open_gripper(' in ext:
    #         gripper.release()
    #         start_of_bracket = ext.index('open_gripper(') + 13
    #         contains_param = ext[start_of_bracket] != ')'
    #         if not contains_param:
    #             done = False
    #             while not done:
    #                 gripper_open = 1.0
    #                 done = gripper.actuate(gripper_open, 0.04)
    #                 self.step()
    #                 self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
    #                 if self._obs_config.record_gripper_closing:
    #                     self._demo_record_step(
    #                         demo, record, callable_each_step)
    #     elif 'close_gripper(' in ext:
    #         start_of_bracket = ext.index('close_gripper(') + 14
    #         contains_param = ext[start_of_bracket] != ')'
    #         if not contains_param:
    #             done = False
    #             while not done:
    #                 gripper_open = 0.0
    #                 done = gripper.actuate(gripper_open, 0.04)
    #                 self.step()
    #                 self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
    #                 if self._obs_config.record_gripper_closing:
    #                     self._demo_record_step(
    #                         demo, record, callable_each_step)

    #     if contains_param:
    #         rest = ext[start_of_bracket:]
    #         num = float(rest[:rest.index(')')])
    #         done = False
    #         while not done:
    #             gripper_open = num
    #             done = gripper.actuate(gripper_open, 0.04)
    #             self.step()
    #             self._joint_position_action = np.append(path.get_executed_joint_position_action(), gripper_open)
    #             if self._obs_config.record_gripper_closing:
    #                 self._demo_record_step(
    #                     demo, record, callable_each_step)

    #     if 'close_gripper(' in ext:
    #         for g_obj in self.task.get_graspable_objects():
    #             gripper.grasp(g_obj)


for i, a in enumerate(action):
    # print(a, a.shape)
    a = np.concatenate([a, np.array([1.])])
    obs, reward, terminate = task.step(a)

print('Done')
env.shutdown()
