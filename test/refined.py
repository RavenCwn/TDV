import os
import glob

def replace_in_files():
    # 查找所有episode目录下的task_related_obj.txt文件
    pattern = "new_data/episode*/data/*/task_related_obj.txt"
    files = glob.glob(pattern)
    
    for file_path in files:
        # 读取文件内容
        with open(file_path, 'r') as file:
            content = file.read()
        
        # 替换文本
        new_content = content.replace('gripper_open', 'gripper_pose')
        
        # 写回文件
        with open(file_path, 'w') as file:
            file.write(new_content)
        
        print(f"已处理文件: {file_path}")

if __name__ == "__main__":
    replace_in_files()