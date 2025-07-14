import numpy as np

# Define the coordinates of the top-left and bottom-right points

# teapot
# top_left = np.array([-0.25, -0.1, 0])
# bottom_right = np.array([0.25, 0.4, 0])

# mug(旧，在中间的)
# top_left = np.array([-0.25, -0.55, 0])
# bottom_right = np.array([0.25, -0.05, 0])

# mug(新，在中间的)
# top_left = np.array([-0.25, -0.5, 0])
# bottom_right = np.array([0.25, 0, 0])

# mug(新，全覆盖)
top_left = np.array([-0.5, -0.75, 0])
bottom_right = np.array([0.5, 0.25, 0])

# mug(一侧的)
# top_left = np.array([-0.25, -0.3, 0])
# bottom_right = np.array([0.25, 0.2, 0])

# holder(一侧)
# top_left = np.array([-0.25, -0.5, 0])
# bottom_right = np.array([0.25, 0, 0])

# steve(中间)
# top_left = np.array([-0.25, -0.05, 0])
# bottom_right = np.array([0.25, 0.45, 0])

# 角度列表
angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

# 运行模式：'spacing' -> 指定间距；'count' -> 指定点数
mode = 'spacing'     # 或 'count'

# spacing 模式下的参数
spacing = 0.04       # 有效时在 x,y 方向上按此间距生成点

# count 模式下的参数
total_points = 50   # 要生成的点的总数（不含角度重复）

# 根据模式生成基础的 (x,y) 点集
if mode == 'spacing':
    x_values = np.arange(top_left[0], bottom_right[0], spacing)
    y_values = np.arange(top_left[1], bottom_right[1], spacing)
    grid_points = np.array([[x, y] for x in x_values for y in y_values])
    
    # 可选：确保包含右下角点
    grid_points = np.vstack([grid_points, bottom_right[:2]])

elif mode == 'count':
    # 计算区域宽高
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    aspect = width / height

    # 根据总点数和长宽比，估算网格行列数
    Nx = max(1, int(np.sqrt(total_points * aspect)))
    Ny = int(np.ceil(total_points / Nx))

    print('NX', Nx)
    print('Ny', Ny)

    # 在 x,y 方向上均匀取样
    x_values = np.linspace(top_left[0], bottom_right[0], Nx)
    y_values = np.linspace(top_left[1], bottom_right[1], Ny)
    mesh = np.array(np.meshgrid(x_values, y_values)).T.reshape(-1, 2)

    # 截取前 total_points 个，确保总数准确
    grid_points = mesh[:total_points]

else:
    raise ValueError("Invalid mode: choose 'spacing' or 'count'")


# 组装最终 offsets（包含角度）
offsets = []
for x, y in grid_points:
    for angle in angles:
        offsets.append([float(x), float(y), 0.0, angle])

# 保存为 txt（Python 列表格式）
with open('offsets_list.txt', 'w') as f:
    f.write(str(offsets))

print(f"Saved {len(offsets)} points to offsets_list.txt (mode={mode})")