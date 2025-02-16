import pickle
import numpy as np

# 读取pkl文件的函数
def load_pkl_file(file_path):
    try:
        with open(file_path, 'rb') as file:  # 以二进制方式打开文件
            data = pickle.load(file)  # 反序列化数据
            print("文件加载成功！")
            return data
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None

# 使用示例
file_path = '/home/a4090/lfwh/trajectory_diff/variation0/episodes/episode0/low_dim_obs.pkl'  # 替换为你的pkl文件路径
file_path = '/home/a4090/lfwh/trajectory_diff/variation0/variation_descriptions.pkl'  # 替换为你的pkl文件路径
file_path = '/home/a4090/lfwh/trajectory_diff/episode0/low_dim_obs.pkl'  # 替换为你的pkl文件路径
file_path = '/home/a4090/lfwh/trajectory_diff/coordiff/data2/close_jar/variation0/episodes/episode0/low_dim_obs.pkl'  # 替换为你的pkl文件路径
file_path = "/home/a4090/lfwh/trajectory_diff/coordiff/mydata/close_jar/variation0/episodes/episode10/low_dim_obs.pkl"

file_path = "/home/c4090/trajectory_diff/coordiff/all_data/close_jar/variation0/episodes/episode51/low_dim_obs.pkl"

data = load_pkl_file(file_path)

if data is not None:
    print("读取的数据：")
    # print(data)
    # import ipdb; ipdb.set_trace()
    obs = data._observations
    for i, o in enumerate(obs):
        objs_pose = o.objs_pose
        # print(objs_pose)
        # print(objs_pose['jar_lid0'])
        # print(task_low_dim_state[k:k+7])
        # print(i, o.grasped_objects, np.linalg.norm((o.gripper_touch_forces)[:3]))
        print(np.linalg.norm((o.gripper_touch_forces)[:3]))