import numpy as np
import argparse
from T import *

def inverse_homogeneous(T):
    """
    计算齐次变换矩阵 T 的逆矩阵。
    T 为 4×4 矩阵，格式为 [R, t; 0, 1]
    返回 T^{-1} = [R^T, -R^T t; 0, 1]
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -np.dot(R.T, t)
    return T_inv

def main(args):
    # 读取外部传入的矩阵
    T1 = np.load(args.T1_path)  # 从命令行传入的 T1 路径(相对相机)
    T2 = np.load(args.T2_path)  # 从命令行传入的 T2 路径(相对相机)

    T1 = T_b_cam_init @ T1[1] 
    T2 = T_b_cam_init @ T2[1]  # 第一帧手部数据有bug

    print("T1:",T1)
    print("T2:",T2)

    # 计算 T1 的逆矩阵
    T1_inv = inverse_homogeneous(T1)

    # 计算相对变换矩阵 T_rel = T1^{-1} * T2
    T_rel = np.dot(T1_inv, T2)

    # 输出并保存最终结果
    print("计算得到的相对变换矩阵 T1 @ T_rel = T2:")
    print("T_o_hand = np.array([")
    for i in range(4):
        row_str = "    [" + ", ".join(f"{T_rel[i, j]:.8f}" for j in range(4)) + "]"
        if i < 3:
            row_str += ","
        print(row_str)
    print("])")

    # 验证结果 T1 @ T_rel = T2
    print(f"验证结果T1 @ T_rel = {np.dot(T1, T_rel)}")

    # 保存计算结果到指定路径
    np.save(args.output_path, T_rel)  # 保存到指定的输出文件路径
    print(f"相对变换矩阵已保存到 {args.output_path}")

if __name__ == '__main__':
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='计算并保存相对变换矩阵')

    # 添加参数
    parser.add_argument('--T1_path', type=str, required=True, help='T1 矩阵的文件路径')
    parser.add_argument('--T2_path', type=str, required=True, help='T2 矩阵的文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='保存结果的输出路径')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(args)
