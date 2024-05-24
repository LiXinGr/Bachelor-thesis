# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import os
import sys
import argparse
from pathlib import Path
import logging
import os
import zipfile
from PIL import Image   
import time

import numpy as np
import cv2
import torch
from tqdm import tqdm
import einops

from MFT.config import load_config
from MFT.point_tracking import convert_to_point_tracking
import MFT.utils.vis_utils as vu
import MFT.utils.io as io_utils
from MFT.utils.misc import ensure_numpy
from scipy.optimize import least_squares
import matplotlib.pyplot as plt



# Hyperparameters
frames_start = 3
frames_end =  33 #23
center_deviation = 0.13
angles_deviation = 0.925  # exp_angle/video0_4_cutter.mp4  #card_angle/video0_186_cutter.mp4
axes_deviation = 0.4
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='', action='store_true')
    parser.add_argument('--gpu', help='cuda device') 
    parser.add_argument('--video', help='path to a source video (or a directory with images)', type=Path,
                        default=Path('demo_in/video0_55_cutter.mp4')) # ugsJtsO9w1A-00.00.24.457-00.00.29.462_HD.mp4
    parser.add_argument('--edit', help='path to a RGBA png with a first-frame edit', type=Path,
                        default=None) # Path('demo_in/edit.png')
    parser.add_argument('--config', help='MFT config file', type=Path, default=Path('configs/MFT_cfg_chain.py'))
    parser.add_argument('--out', help='output directory', type=Path, default=Path('demo_out/'))
    parser.add_argument('--arrays', help='arrays directory', type=Path, default=Path('demo_arrays/'))
    parser.add_argument('--grid_spacing', help='distance between visualized query points', type=int, default= 30)
    parser.add_argument('--lineall', help='Paint trajectory for all points', type=bool, default=False)
    parser.add_argument('--lineone', help='Paint trajectory for one point', type=bool, default=False)
    parser.add_argument('--ellipses', help='Paint ellipses for all point', type=bool, default=False)
    parser.add_argument('--lineonenumber', help='Paint trajectory for all points', type=int, default=1)


    args = parser.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl, format=format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return args

def run(args):  
    start_time = time.time()
    config = load_config(args.config)
    logger.info("Loading tracker")
    tracker = config.tracker_class(config)
    logger.info("Tracker loaded")

    initialized = False
    queries = None

    results = []

    logger.info("Starting tracking")

    video_path = args.video
    output_zip_path = 'frames.zip'  # Path for the output zip file

    video_name = args.video.stem
    dir_path = args.arrays / f'{video_name}'
    dir_path.mkdir(parents=True, exist_ok=True)

    """
    # Create a zip file for writing
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        # Iterate over frames and add them to the zip file
        for frame_idx, frame in enumerate(tqdm(io_utils.get_video_frames(video_path), total=io_utils.get_video_length(video_path))):
            image = Image.fromarray(frame)

            # Save each image to a temporary file
            temp_image_path = f'temp_image_{frame_idx}.jpg'
            image.save(temp_image_path)

            # Add the temporary image file to the zip file
            zipf.write(temp_image_path, os.path.basename(temp_image_path))

            # Remove the temporary image file
            os.remove(temp_image_path)

    print("Frames have been zipped successfully.")
    """
    
    # 2D list where one list is the list of coordinates of all points in the current frame
    coords_list = []
    # the list of frames
    frames_list = []

    for frame in tqdm(io_utils.get_video_frames(args.video), total=io_utils.get_video_length(args.video)):
        if not initialized:
            meta = tracker.init(frame)
            initialized = True
            queries = get_queries(frame.shape[:2], args.grid_spacing)
        else:
            meta = tracker.track(frame)
        coords, occlusions = convert_to_point_tracking(meta.result, queries)
        coords_list.append(coords)
        frames_list.append(frame)
        result = meta.result
        result.cpu()
        results.append((result, coords, occlusions))

    print("coords_list: ")
    print(len(coords_list))
    print(len(coords_list[0]))
    ##########################################################################################
    
    center_list, coords_res, sort_res_by_all, mean_center, ellipse_list, ellipse = function(args, coords_list)

    ###########################################################################################
    
    
    edit = None
    #if args.edit.exists():
        #edit = cv2.imread(str(args.edit), cv2.IMREAD_UNCHANGED)

    N = coords.shape[0]
    # Generate a sequence of colors
    values = np.linspace(0, 1, N)  # Generate evenly spaced values between 0 and 1
    # Choose a colormap
    colormap = plt.get_cmap('turbo')
    normalized_indices = np.linspace(0, 1, N)
    # Map values to colors in the colormap
    num_colors = N  # Assuming N is the number of dots
    colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(num_colors)]

    logger.info("Drawing the results")
    video_name = args.video.stem
    if args.lineall:
        line_all_writer = vu.VideoWriter(args.arrays / f'{video_name}'/ f'{video_name}_line_all.mp4', fps=15, images_export=False)
    if args.lineone:
        line_writer = vu.VideoWriter(args.arrays / f'{video_name}'/ f'{video_name}_line.mp4', fps=15, images_export=False)
    point_writer = vu.VideoWriter(args.out / f'{video_name}_points.mp4', fps=15, images_export=False)
    if edit is not None:
        edit_writer = vu.VideoWriter(args.out / f'{video_name}_edit.mp4', fps=15, images_export=False)
    for frame_i, frame in enumerate(tqdm(io_utils.get_video_frames(args.video),
                                         total=io_utils.get_video_length(args.video))):
        result, coords, occlusions = results[frame_i]

        dot_vis = draw_dots(frame, coords, occlusions, colormap, normalized_indices)
        if args.lineone:
            line_vis = draw_line_one(frame, args.lineonenumber, coords_list, colormap, normalized_indices, center_list[args.lineonenumber], frame_i)
            line_writer.write(line_vis)
        if args.lineall:
            line_all = draw_line_all(frame, coords_list, colormap, normalized_indices, frame_i)
            line_all_writer.write(line_all)



        if edit is not None:
            edit_vis = draw_edit(frame, result, edit)
        if False:
            cv2.imshow("cv: dot vis", dot_vis)
            while True:
                c = cv2.waitKey(0)
                if c == ord('q'):
                    sys.exit(1)
                elif c == ord(' '):
                    break

        point_writer.write(dot_vis)
            
        if edit is not None:
            edit_writer.write(edit_vis)
    point_writer.close()
    if edit is not None:
        edit_writer.close()
    if args.lineall:
        line_all_writer.close()
    if args.lineone:
        line_writer.close()
    
    
    """
    mean_degrees = []
    for co in coords_res:
        angles = []
        for i in range(0, 19):
            angle = find_angle(co[i], center, co[i+1])
            angles.append(angle)
        np_angles = np.array(angles)
        mean_degrees.append(np_angles)
    np_mean_degrees = np.array([arr for arr in mean_degrees])
    degree = np.mean(np_mean_degrees)
    rotation_speed = degree * 30 * 60 / 360
    print("Rotation speed:", rotation_speed, "(rpm)")
    """

    matrix = np.array([[1/ellipse[4]**2, 0], [0, 1/ellipse[5]**2]])

    print(matrix)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    P = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    
    print(P)
    print(P.shape)
    print(coords_res.shape)

    coords_list = np.array([arr for arr in coords_list])
    coords_list_resgaoed = coords_list.reshape(-1,2)
    coords_res_reshaped = coords_res.reshape(-1,2)
    trans = np.dot(coords_res_reshaped - ellipse[:2], P.T) + ellipse[:2]
    tr = coords_list_resgaoed @ P.T

    coords_res = trans.reshape(coords_res.shape)
    coords_list = tr.reshape(coords_list.shape)



    mean_center = np.dot(mean_center - ellipse[:2], P.T) + ellipse[:2]

    
    mean_degrees = []
    np_an = []
    k = 0
    print(coords_res.shape)

    for co in coords_res:
        angles = np.array([])
        center, radius = fit_circle(co[frames_start:frames_end])
        for i in range(frames_start, frames_end):
            angle = find_angle(co[i], mean_center, co[i+1])
            angles = np.append(angles, np.round(angle, decimals=1))
        if k == 53:
            print(angles)
            print(center)
        k += 1
        #mean_degrees.append(np.mean(np.where(angles > np.mean(angles))))
        np_an.append(angles)
        mean_degrees.append(np.mean(angles[angles > angles_deviation * np.median(angles)]))
    np_mean_degrees = np.array([arr for arr in mean_degrees])
    np_mean_degrees = np_mean_degrees[~np.isnan(np_mean_degrees)]
    print("The mean degree:", np_mean_degrees)
    #degree = np.median(np_mean_degrees)
    print("Angles:")
    print(np.vstack(np_an))
    degree = np.mean(np_mean_degrees[np_mean_degrees > angles_deviation * np.median(np_mean_degrees)])


    print(degree)
    rotation_speed = degree * 5 # 30 * 60 / 360
    print("Rotation speed:", rotation_speed, "(rpm)")


    file_np_mean_degrees = dir_path / "np_mean_degrees.npy"
    file_np_an = dir_path / "np_an.npy"

    np.save(file_np_mean_degrees, np_mean_degrees)
    np.save(file_np_an, np_an)


    if args.lineall:
        line_all_writer = vu.VideoWriter(args.arrays / f'{video_name}'/ f'{video_name}_line_all_trans.mp4', fps=15, images_export=False)
    if args.lineone:
        line_writer = vu.VideoWriter(args.arrays /f'{video_name}'/ f'{video_name}_line_trans.mp4', fps=15, images_export=False)

    for frame_i, frame in enumerate(tqdm(io_utils.get_video_frames(args.video),
                                         total=io_utils.get_video_length(args.video))):
        result, coords, occlusions = results[frame_i]


        if args.lineone:
            line_vis = draw_line_one(frame, args.lineonenumber, coords_list, colormap, normalized_indices, center_list[args.lineonenumber], frame_i)
            line_writer.write(line_vis)
        if args.lineall:
            line_all = draw_line_all(frame, coords_list, colormap, normalized_indices, frame_i)
            line_all_writer.write(line_all)

    if args.lineall:
        line_all_writer.close()
    if args.lineone:
        line_writer.close()


    

    # draw trajectory
    if args.lineone:
        fr = frames_list[frames_end].copy()
        
        for i in range(0, sort_res_by_all.shape[0]):
            vu.circle(fr, sort_res_by_all[i][:2], sort_res_by_all[i][2], colors[sort_res_by_all[i][3].astype(np.int32)])
        
        # Save the image using Pillow
        #vu.circle(fr, center_list[args.lineonenumber], radius_list[args.lineonenumber], colors[args.lineonenumber])
        image = Image.fromarray(fr)
        image.save(args.arrays /f'{video_name}'/  f'{video_name}_lse.jpg')

        fr = frames_list[frames_end].copy()

        
        for i in range(0, ellipse_list.shape[0]):
            cv2.ellipse(fr, center=np.round(np.maximum(ellipse_list[i][:2], 0)).astype(np.int32), axes=np.round(np.maximum(ellipse_list[i][2:4], 0)).astype(np.int32), angle=ellipse_list[i][4], startAngle=0, endAngle=360, color=colors[ellipse_list[i][5].astype(np.int32)])
        

        # Save the image using Pillow
        #vu.circle(fr, center_list[args.lineonenumber], radius_list[args.lineonenumber], colors[args.lineonenumber])
        image = Image.fromarray(fr)
        image.save(args.arrays /f'{video_name}'/ f'{video_name}_ellipses.jpg')

        fr = frames_list[frames_end].copy()

        cv2.ellipse(fr, center=ellipse[:2].astype(np.int32), axes=ellipse[4:6].astype(np.int32), angle=ellipse[6], startAngle=0, endAngle=360, color=(255,100,100))
        image1 = Image.fromarray(fr)
        image1.save(args.arrays /f'{video_name}'/ f'{video_name}_one_ellipse.jpg')


    end_time = time.time()
    elapsed_time = end_time - start_time



    print("Elapsed time:", elapsed_time, "seconds")
    return 0

# TODO: add a description to this function
def function(args, coords_list):
    # a list of the centers of the corresponding best circles
    center_list = []
    # ---||---  the radii --------------||-------------------
    radius_list = []
    # 2D list where each element is a list of (center_x, center_y, radius, index)
    result_list = []
    result_list_ellipse = []
    array_of_arrays = np.array([arr for arr in coords_list])
    ellipses_list = []
    a_focus_list = []
    b_focus_list = []
    theta_list = []

    arr_swapped = np.transpose(array_of_arrays, (1, 0, 2))   #(1032, 217, 2)  (N_points, N_frames, 2)
    # create lists of centers and radii for each point using the least square error
    for i in range(0, arr_swapped.shape[0]):
        mean_center, radius = fit_circle(arr_swapped[i][frames_start:frames_end])
        #mean_center, axes, theta = fit_ellipse(arr_swapped[i][frames_start:frames_end])
        #ellipse = func(arr_swapped[i][frames_start:frames_end])
        a,b,c = fit_ellipse(arr_swapped[i][frames_start:frames_end])
        a = tuple(round(value, 2) for value in mean_center)
        ellipse = (a,b,c)
        mean_center[0] = a[0]
        mean_center[1] = a[1]

        a_focus_list.append(b[0])
        b_focus_list.append(b[1])
        theta_list.append(c)
        result_list.append(np.append(mean_center, [radius, i]))
        result_list_ellipse.append(np.append(mean_center, [radius, i, b[0], b[1], c]))
        center_list.append(mean_center)
        radius_list.append(radius)
        ellipses_list.append((ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2], i))


        

    result_list = result_list_ellipse
    # create np.array from lists
    np_theta_list = np.array([arr for arr in theta_list])
    np_a_focus_list = np.array([arr for arr in a_focus_list])
    np_b_focus_list = np.array([arr for arr in b_focus_list])
    np_ellipse_list = np.round(np.array([arr for arr in ellipses_list]), decimals=2)
    np_center_list = np.array([arr for arr in center_list])
    np_radius_list = np.array([arr for arr in radius_list])
    np_result_list = np.array([arr for arr in result_list])# the list of (center_x, center_y, radius, index)
    np_result_list = np.round(np_result_list, decimals=2)
    # delete noise using the mean radius
    sort_res_by_radius = np_result_list #[(np_result_list[:,2] > median_radius - circule_deviation * median_radius) & (np_result_list[:,2] < median_radius + circule_deviation * median_radius)] # (N_points, 4)
    # find mean radius and mean center without noise
    sort_res_mean = np.round(np.median(np_result_list, axis=0), decimals=2)

    print(sort_res_mean)
    # delete noise using the median center
    sort_res_by_radius = sort_res_by_radius[(sort_res_by_radius[:,1] > sort_res_mean[1] - center_deviation * sort_res_mean[1]) 
                                         & (sort_res_by_radius[:,1] < sort_res_mean[1] + center_deviation * sort_res_mean[1]) 
                                         & (sort_res_by_radius[:,0] > sort_res_mean[0] - center_deviation * sort_res_mean[0]) 
                                         & (sort_res_by_radius[:,0] < sort_res_mean[0] + center_deviation * sort_res_mean[0])
                                         #& (sort_res_by_radius[:,4] > sort_res_mean[4] - 0.5 * sort_res_mean[4])
                                         #& (sort_res_by_radius[:,4] < sort_res_mean[4] + 0.5 * sort_res_mean[4])
                                         #& (sort_res_by_radius[:,5] > sort_res_mean[5] - 0.5 * sort_res_mean[5])
                                         #& (sort_res_by_radius[:,5] < sort_res_mean[5] + 0.5 * sort_res_mean[5])
                                         ]
    
    print("Sort res by all radius:")
    print(sort_res_by_radius.shape)

    indexes1 = sort_res_by_radius[:,3].astype(np.int32)
    
    sort_res_mean = np.round(np.median(sort_res_by_radius, axis=0), decimals=2)

    axes_ration = np.round(np.median(sort_res_by_radius[:,4] / sort_res_by_radius[:, 5]), decimals=2)

    """
    sort_res_by_all = sort_res_by_radius[(sort_res_by_radius[:,4] > sort_res_mean[4] - 0.5 * sort_res_mean[4])
                                         & (sort_res_by_radius[:,4] < sort_res_mean[4] + 0.5 * sort_res_mean[4])
                                         & (sort_res_by_radius[:,5] > sort_res_mean[5] - 0.5 * sort_res_mean[5])
                                         & (sort_res_by_radius[:,5] < sort_res_mean[5] + 0.5 * sort_res_mean[5])
                                         ] 
    """

    sort_res_by_all = sort_res_by_radius[(sort_res_by_radius[:,4] / sort_res_by_radius[:,5] > axes_ration - axes_deviation * axes_ration)
                                         & (sort_res_by_radius[:,4] / sort_res_by_radius[:,5] < axes_ration + axes_deviation * axes_ration)
                                         #& (sort_res_by_radius[:,6] > sort_res_mean[6] - 0.5 * sort_res_mean[6])
                                         #& (sort_res_by_radius[:,6] < sort_res_mean[6] + 0.5 * sort_res_mean[6])
                                         ]  
    
                                                                   
    print("Sort res by all axes:")
    print(sort_res_by_all.shape)
    
    sort_res_mean = np.round(np.mean(sort_res_by_all, axis=0), decimals=2)
    print(sort_res_mean)
    mean_radius = sort_res_mean[2]
    mean_center = sort_res_mean[:2]
    mean_a = sort_res_mean[4]
    mean_b = sort_res_mean[5]
    mean_theta = sort_res_mean[6]
    
    # delete noise using the median radius
    #sort_res_by_all = sort_res_by_all[(sort_res_by_all[:,2] > sort_res_mean[2] - circule_deviation * sort_res_mean[2])
                                      #& (sort_res_by_all[:,2] < sort_res_mean[2] + circule_deviation * sort_res_mean[2])
    #                                  ]
    # take indexes of the remaining points
    print("The best points(center_x, center_y, radius, index):")
    indexes = sort_res_by_all[:,3].astype(np.int32)
    # take coordinates of remaining points through the video    
    coords_res = arr_swapped[indexes] #(165, 217, 2)
    # TODO: now we have 2 options: (find an angle between coords) and (find the nearest points on the lse circule and after then find an angle between coords)
    # The first way:
    # Create the full path to the subdirectory
    video_name = args.video.stem
    dir_path = args.arrays / f'{video_name}'
    dir_path.mkdir(parents=True, exist_ok=True)

    file_np_result_list = dir_path / "np_result_list.npy"
    file_center = dir_path / "mean_center.npy"
    file_radius = dir_path / "mean_radius.npy"
    file_sort_res_by_all = dir_path / "sort_res_by_all.npy"
    file_np_ellipses_list = dir_path / "np_ellipses_list.npy"

    np.save(file_np_result_list, np_result_list)
    np.save(file_center, mean_center)
    np.save(file_radius, mean_radius)
    np.save(file_sort_res_by_all, sort_res_by_all)
    np.save(file_np_ellipses_list, np_ellipse_list)

    return center_list, coords_res, sort_res_by_all, mean_center, np_ellipse_list[indexes], sort_res_mean



def find_angle(first, center, second):
    # Calculate vectors between points
    v1 = first - center
    v2 = second - center

    # Calculate dot product
    dot_product = np.dot(v1, v2)

    # Calculate magnitudes of vectors
    v1_magnitude = np.linalg.norm(v1)
    v2_magnitude = np.linalg.norm(v2)

    # Calculate cosine of angle between vectors
    cos_angle = dot_product / (v1_magnitude * v2_magnitude)

    # Calculate angle in radians
    angle_rad = np.arccos(cos_angle)

    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def get_queries(frame_shape, spacing):
    H, W = frame_shape
    xs = np.arange(0, W, spacing)
    ys = np.arange(0, H, spacing)

    xs, ys = np.meshgrid(xs, ys)
    flat_xs = xs.flatten()
    flat_ys = ys.flatten()

    queries = np.vstack((flat_xs, flat_ys)).T
    return torch.from_numpy(queries).float().cuda()

def draw_dots(frame, coords, occlusions,colormap, indexes):
    canvas = frame.copy()
    N = coords.shape[0] #(1032,2)
    # [(30, 0), (60, 0) ... (660, 1260), (690, 1260)]
    

    for i in range(N):
        occl = occlusions[i] > 0.5
        #thickness = 1 if occl else -1
        color_float = colormap(indexes[i])[:3]
        color = tuple(int(x*255) for x in color_float)
        
        vu.circle(canvas, coords[i, :], radius=3, color=color, thickness=-1)

    return canvas

def draw_line_one(frame, j, coords_list, colormap, indexes, center, frame_i):   
    canvas = frame.copy()
    N = coords_list[0].shape[0]

    coords = np.transpose(np.array([arr for arr in coords_list]), axes=(1,0,2))

    color_float = colormap(indexes[j])[:3]
    color = tuple(int(x*255) for x in color_float)
    vu.circle(canvas, coords_list[frame_i][j], radius=4, color=color, thickness=-1)
    #vu.circle(canvas, center, radius=3, color=color, thickness=-1)
    ellipse = fit_ellipse(coords[j][frames_start:frames_end])
    arr = np.array([ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2]])

    circle_x, circle_y = int(coords_list[0][j][0]), int(coords_list[0][j][1])
    text = f"({circle_x}, {circle_y})"
            
    # Font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)  # White color
    
    # Calculate position for text
    text_position = (circle_x, circle_y - 10)  # Just above the circle
    
    # Draw text on canvas
    #cv2.putText(canvas, text, text_position, font, font_scale, text_color, font_thickness)

    for i in range(0,frame_i):
        vu.circle(canvas, coords_list[i][j], radius=4, color=color, thickness=-1)
        vu.line(canvas, coords_list[i][j], coords_list[i+1][j], color=color, thickness=3)
        vu.circle(canvas, coords_list[i+1][j], radius=4, color=color, thickness=-1)
        circle_x, circle_y = int(coords_list[i+1][j][0]), int(coords_list[i+1][j][1])
        text = f"({circle_x}, {circle_y})"
            
        # Font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (255, 255, 255)  # White color
        
        # Calculate position for text
        text_position = (circle_x, circle_y - 10)  # Just above the circle
        
        # Draw text on canvas
        #cv2.putText(canvas, text, text_position, font, font_scale, text_color, font_thickness)        
        #cv2.ellipse(canvas, center=np.round(arr[:2]).astype(np.int32), axes=np.round(arr[2:4]).astype(np.int32), angle=arr[4], startAngle=0, endAngle=360, color=color)
        


    return canvas

def draw_line_all(frame, coords_list, colormap, indexes, frame_i):
    canvas = frame.copy()
    N = coords_list[0].shape[0]


    for j in range(N):
        color_float = colormap(indexes[j])[:3]
        color = tuple(int(x*255) for x in color_float)
        vu.circle(canvas, coords_list[frame_i][j], radius=3, color=color, thickness=-1)
        for i in range(0, frame_i):
            vu.line(canvas, coords_list[i][j], coords_list[i+1][j], color=color)


    return canvas

def draw_edit(frame, result, edit):
    occlusion_in_template = result.occlusion
    template_visible_mask = einops.rearrange(occlusion_in_template, '1 H W -> H W') < 0.5
    template_visible_mask = template_visible_mask.cpu()
    edit_mask = torch.from_numpy(edit[:, :, 3] > 0)
    template_visible_mask = torch.logical_and(template_visible_mask, edit_mask)

    edit_alpha = einops.rearrange(edit[:, :, 3], 'H W -> H W 1').astype(np.float32) / 255.0
    premult = edit[:, :, :3].astype(np.float32) * edit_alpha
    color_transfer = ensure_numpy(result.warp_forward(premult, mask=template_visible_mask))
    color_transfer = np.clip(color_transfer, 0, 255).astype(np.uint8)
    alpha_transfer = ensure_numpy(result.warp_forward(
        einops.rearrange(edit[:, :, 3], 'H W -> H W 1'),
        mask=template_visible_mask
    ))
    vis = vu.blend_with_alpha_premult(color_transfer, vu.to_gray_3ch(frame), alpha_transfer)
    return vis


def circle_residuals(params, points):
    # Unpack parameters
    cx, cy, r = params

    # Compute distances from points to circle
    distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)

    # Compute squared residuals
    residuals = (distances - r)**2

    return residuals

def fit_circle(points):
    # Initial estimates for circle parameters
    initial_center = np.mean(points, axis=0)
    initial_radius = np.mean(np.sqrt(np.sum((points - initial_center)**2, axis=1)))

    initial_params = [initial_center[0], initial_center[1], initial_radius]

    # Use least squares optimization to fit the circle
    result = least_squares(circle_residuals, initial_params, args=(points,))
    
    # Extract fitted circle parameters
    center = result.x[:2]
    radius = result.x[2]

    return center, radius

def ellipse_residuals(params, points):
    # Unpack parameters
    cx, cy, a, b, theta = params

    # Compute distances from points to ellipse
    x_diff = points[:, 0] - cx
    y_diff = points[:, 1] - cy
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    distances = ((cos_theta * x_diff + sin_theta * y_diff) / a)**2 + ((-sin_theta * x_diff + cos_theta * y_diff) / b)**2

    # Compute squared residuals
    residuals = distances - 1

    return residuals

def fit_ellipse(points):
    # Initial estimates for ellipse parameters
    initial_center = np.mean(points, axis=0)
    initial_axes = np.std(points, axis=0)
    initial_theta = 0.0

    initial_params = [initial_center[0], initial_center[1], initial_axes[0]+1.0e-07, initial_axes[1]+1.0e-07, initial_theta]

    # Use least squares optimization to fit the ellipse
    result = least_squares(ellipse_residuals, initial_params, args=(points,))
    
    # Extract fitted ellipse parameters
    center = result.x[:2]
    axes = result.x[2:4]
    theta = result.x[4]

    return center, axes, theta

from ipdb import iex
@iex
def main():
    args = parse_arguments()
    torch.cuda.set_device(3)
    return run(args)


if __name__ == '__main__':
    results = main()
