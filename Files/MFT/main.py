import numpy as np
import matplotlib.pyplot as plt 
import os
from pathlib import Path
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_name', help="The name of the video file", type=str, default="video0_6_cutter")
    
    args = parser.parse_args()

    return args

args = parse_arguments()
video_name = args.video_name

data_result_list = np.load("demo_arrays/" + video_name + "/np_result_list.npy")
x_res_list = data_result_list[:, 0]
y_res_list = data_result_list[:, 1]

data_center = np.load("demo_arrays/" + video_name + "/mean_center.npy")
x_mean_center = data_center[0]
y_mean_center = data_center[1]

dir_path = Path("matplotlib_results/" + video_name)
dir_path.mkdir(parents=True, exist_ok=True)
# Create a scatter plot
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
plt.scatter(x_res_list, y_res_list, s=10, color='blue', alpha=0.5)  # Adjust marker size and color
#plt.scatter(x_mean_center, y_mean_center, s=10, color='red', alpha=0.5)
if video_name == "video0_134_ctter":
    plt.scatter(367.5, 380.5, s=10, color='green', alpha=0.5)
    plt.text(367.5, 380.5, f'({367.5}, {380.5})', fontsize=10, ha='left', color='g') # Add text
#else:
    #plt.scatter(408.5, 364, s=10, color='green', alpha=0.5)
    #plt.text(408.5, 364, f'({408.5}, {364})', fontsize=10, ha='left', color='g') # Add text
#plt.text(x_mean_center, y_mean_center, f'({x_mean_center}, {y_mean_center})', fontsize=10, ha='right', color='r') # Add text
plt.title('Distribution of centers') # Add title
plt.xlabel('X Coordinate')  # Add x-axis label
plt.ylabel('Y Coordinate')  # Add y-axis label
plt.grid(True)  # Add grid
plt.savefig(dir_path / "centers_distribution.png")
plt.close()


data_radius_list = data_result_list[:, 2] #np.load("demo_arrays/" + video_name + "/np_radius_list.npy")
x_values = np.arange(len(data_radius_list))

data_mean_radius = np.load("demo_arrays/" + video_name + "/mean_radius.npy")

# Plot the data as a scatter plot
plt.scatter(x_values, data_radius_list)
plt.axhline(y=data_mean_radius, color='r', linestyle='--', label="Line at {}".format(data_mean_radius))
#plt.axhline(y=np.median(data_radius_list), color='g', linestyle='--', label="Line at {}".format(np.median(data_radius_list)))
# Mark the specific value with a text annotation
plt.text(0, data_mean_radius, "Value: {}".format(data_mean_radius), color='r', fontsize=10, verticalalignment='bottom')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Radii distribution')
plt.savefig(dir_path / "radii_distribution.png")
plt.close()


data_sort_res_by_radius = np.load("demo_arrays/" + video_name + "/sort_res_by_all.npy")
data_sorted_radii = data_sort_res_by_radius[:,2]

x_values = np.arange(len(data_sorted_radii))

# Plot the data as a scatter plot
plt.scatter(x_values, data_sorted_radii)
plt.axhline(y=data_mean_radius, color='r', linestyle='--', label="Line at {}".format(data_mean_radius))
#plt.axhline(y=np.median(data_radius_list), color='g', linestyle='--', label="Line at {}".format(np.median(data_radius_list)))
# Mark the specific value with a text annotation
plt.text(0, data_mean_radius, "Value: {}".format(data_mean_radius), color='r', fontsize=10, verticalalignment='bottom')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Sorted radii distribution')
plt.savefig(dir_path / "sorted_radii_distribution.png")
plt.close()


"""

data_np_mean_degrees = np.load("demo_arrays/" + video_name + "/np_an.npy")
#f = data_np_mean_degrees[data_np_mean_degrees > np.median(data_np_mean_degrees) * 1.2]
x_values = np.arange(len(data_np_mean_degrees))
# Plot the data as a scatter plot
print(np.mean(data_np_mean_degrees[data_np_mean_degrees > np.median(data_np_mean_degrees)]))
plt.scatter(x_values, data_np_mean_degrees)
plt.axhline(y=np.mean(data_np_mean_degrees[data_np_mean_degrees > np.median(data_np_mean_degrees)]), color='r', linestyle='--', label="Line at {}".format(data_mean_radius))
#plt.axhline(y=np.median(data_radius_list), color='g', linestyle='--', label="Line at {}".format(np.median(data_radius_list)))
# Mark the specific value with a text annotation
#plt.text(0, data_mean_radius, "Value: {}".format(data_mean_radius), color='r', fontsize=10, verticalalignment='bottom')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Sorted degrees distribution')
plt.savefig(dir_path / "np_mean_degrees_distribution.png")
plt.close()

"""
data_np_mean_degrees = np.load("demo_arrays/" + video_name + "/np_an.npy")
N, k = data_np_mean_degrees.shape
# Plot each row as a separate line plot
for i in range(N):
    plt.scatter(np.full(k, i), data_np_mean_degrees[i], label=f'Dataset {i}', s=5, color='b')

plt.axhline(y=18, color='r', linestyle='--', label="Line at {}".format(data_mean_radius), linewidth=2)

# Set labels and title
plt.xlabel('The index of the selected point')
plt.ylabel('The angle value')
plt.title('Distribution of angle values')

# Show legend
#plt.legend()
plt.savefig(dir_path / "np_mean_degrees_distribution.png")
plt.close()

# Show plot
plt.show()










