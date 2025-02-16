import numpy as np
from glob import glob

data_dirs = "/home/a4090/lfwh/trajectory_diff/coordiff/data3_re/close_jar/train"

for data_dir in sorted(glob(data_dirs+'/*')):
    action_0 = data_dir + "/0" + '/relevant_traj.npy'
    action_1 = data_dir + "/1" + '/relevant_traj.npy'

    action_0 = np.load(action_0)
    action_1 = np.load(action_1)
    print(action_0.shape)
    print(action_1.shape)