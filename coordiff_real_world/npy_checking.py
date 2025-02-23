import numpy as np

npy_file = 'episode_0/cup0_poses.npy'  
array = np.load(npy_file)

txt_file = 'output.txt'  

with open(txt_file, 'w') as f:
    for i in range(array.shape[0]):  
        f.write(f"切片 {i}:\n")  
        np.savetxt(f, array[i], fmt='%s')  
        f.write("\n")  

print(f"已保存到 {txt_file}")
