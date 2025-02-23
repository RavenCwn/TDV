import os
import glob
from natsort import natsorted
import imageio

# 指定根目录
root_dir = "video_save/"

# 遍历根目录下的所有子目录
task_name = os.listdir(root_dir)

for i, var in enumerate(task_name):
    path = os.path.join(root_dir, var)
    front_rgb_path = os.path.join(path, "front_rgb")
    
    # 构建 PNG 文件的路径模式
    png_pattern = os.path.join(front_rgb_path, "*.png")
    
    # 使用 natsorted 对 PNG 文件进行自然排序
    png_files = natsorted(glob.glob(png_pattern))
    
    if not png_files:
        print(f"警告: 在路径 {png_pattern} 中未找到 PNG 文件。")
        continue
    
    # 初始化视频写入器
    output_filename = f"{var}.mp4"
    fps = 30  # 设置帧率
    with imageio.get_writer(output_filename, fps=fps) as writer:
        for png_file in png_files:
            image = imageio.imread(png_file)
            writer.append_data(image)
    
    print(f"已成功生成视频: {output_filename}")