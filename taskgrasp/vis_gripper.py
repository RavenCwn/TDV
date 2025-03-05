import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Define the transformation matrices
# T_cam1_to_gripper = np.array([
#     [ 0.01383618,  0.99913825, -0.03913196, -0.06690409],
#     [-0.99978003,  0.01444079,  0.01521018,  0.03126657],
#     [ 0.01576217,  0.0389129 ,  0.99911828, -0.18269207],
#     [ 0.0,          0.0,          0.0,          1.0]
# ])
# T_gripper_to_base = np.array([
#     [ 0.54188514,  0.57089991, -0.61679315, -0.13319252 ],
#     [ 0.46443295, -0.8150583 , -0.34638419, -0.21292606],
#     [-0.70047307, -0.09875862, -0.70681271,  0.3352544 ],
#     [ 0. , 0. , 0. , 1. ]
# ])
# T_cam1_to_base = T_gripper_to_base @ T_cam1_to_gripper
# print("T_cam1_to_base: ",  T_cam1_to_base)


T_cam2_to_base = np.array([
    # [0.70437283, -0.48266104, 0.52047789, -0.9106794],
    # [-0.70976469, -0.4888679, 0.50719056, -0.86339475],
    # [0.00964381, -0.72666808, -0.68692103, 0.42432687],
    # [0., 0., 0., 1.]
     [0.65617483, -0.3394302, 0.67395974, -0.90763576],
    [-0.74992243, -0.39270395, 0.53235323, -0.85760725],
    [0.08396989, -0.85473431, -0.51222877, 0.41725306],
    [0.0, 0.0, 0.0, 1.0]
])

T_cam1_to_gripper = np.array([
[-0.02869299,  0.99947079,  0.01532479, -0.07960934],
[-0.99904825, -0.02817024, -0.03330199,  0.04868053],
[-0.03285266, -0.01626574,  0.99932784, -0.16906925],
[0.0, 0.0, 0.0, 1.0]
])
T_gripper_to_base = np.array([
    # [ 0.54188514,  0.57089991, -0.61679315, -0.13319252 ],
    # [ 0.46443295, -0.8150583 , -0.34638419, -0.21292606],
    # [-0.70047307, -0.09875862, -0.70681271,  0.3352544 ],
    # [ 0. , 0. , 0. , 1. ]
    [0.70188554, 0.71191884, -0.02298362, -0.23082184],
    [0.71213774, -0.70070374, 0.04329108, -0.25256987],
    [0.01471503, -0.04675289, -0.9987981, 0.33429483],
    [0.0, 0.0, 0.0, 1.0]
])

T_gripper_to_base = np.array([[ 0.54178783,  0.57080654, -0.61696502,-0.13318502],
    [ 0.46410755, -0.81514886, -0.34660715,-0.21293023],
    [-0.70076396, -0.09855059, -0.70655336, 0.3353847],
    [0,0,0,1]])
T_cam1_to_base = T_gripper_to_base @ T_cam1_to_gripper

# Function to extract the position and orientation from the transformation matrix
def extract_position_orientation(T):
    position = T[:3, 3]
    orientation = T[:3, :3]
    return position, orientation

# Extract positions and orientations
pos_base, rot_base = extract_position_orientation(np.eye(4))  # World frame (base)
pos_gripper, rot_gripper = extract_position_orientation(T_gripper_to_base)
pos_cam1, rot_cam1 = extract_position_orientation(T_cam1_to_base)
pos_cam2, rot_cam2 = extract_position_orientation(T_cam2_to_base)

# Create a figure for the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define scale for arrows
scale = 0.1

# Function to plot the coordinate axes based on a rotation matrix
def plot_frame(position, rotation, label, ax, scale=0.1):
    ax.quiver(position[0], position[1], position[2], rotation[0, 0], rotation[1, 0], rotation[2, 0], length=scale, color='r')
    ax.quiver(position[0], position[1], position[2], rotation[0, 1], rotation[1, 1], rotation[2, 1], length=scale, color='g')
    ax.quiver(position[0], position[1], position[2], rotation[0, 2], rotation[1, 2], rotation[2, 2], length=scale, color='b')
    ax.text(position[0], position[1], position[2], label, color='black')

# Plotting the world frame (Base), Gripper, Camera 1, and Camera 2
plot_frame(pos_base, rot_base, 'World Frame', ax, scale)
plot_frame(pos_gripper, rot_gripper, 'Gripper Frame', ax, scale)
plot_frame(pos_cam1, rot_cam1, 'Camera 1 Frame', ax, scale)
plot_frame(pos_cam2, rot_cam2, 'Camera 2 Frame', ax, scale)

# Set limits for the plot
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('World Frame, Gripper Frame, Camera 1 and Camera 2')

# Show the plot
plt.show()
