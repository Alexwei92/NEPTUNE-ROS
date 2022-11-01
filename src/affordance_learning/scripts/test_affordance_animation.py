"""
Test Affordance Net Performance
"""
import os, glob
import pickle
import pandas
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

import rospy
from sensor_msgs.msg import Image, CompressedImage

from utils.plot_utils import FieldMapPlot
from controller import AffordanceCtrl


BLACK  = (0,0,0)
WHITE  = (255,255,255)
RED    = (0,0,255)
BLUE   = (255,0,0)
GREEN  = (0,255,0)
YELLOW = (0,255,255)
CYAN   = (255,255,0)

# class TestAffordance():

#     def __init__(self):
#         rospy.init_node("test_affordance")
#         self.define_subscriber()

#     def define_subscriber(self):
#         color_image_topic = "/d435i/color/image_raw/compressed"
#         rospy.Subscriber(
#             color_image_topic,
#             CompressedImage,
#             self.color_image_callback,
#             queue_size=10,
#         )

#     def 


if __name__ == "__main__":
    curr_dir = os.path.dirname(__file__)
    data_path = os.path.join(curr_dir, "ground_truth/plant_field.pkl")

    ### Configure map
    with open(data_path, 'rb') as file:
        data = pickle.load(file)

    field_bound = {
        'latlon': data['field_bound_latlon'],
        'local': data['field_bound_local'],
    }

    # map_handler = FieldMapPlot(
    #     data['row_data'],
    #     field_bound,
    # )

    print('Load field data successfully!')

    ### Load Network
    model_param = {
        'afford_model_path': '/media/lab/NEPTUNE2/field_outputs/row_10/affordance/affordance_model.pt'
    }
    
    agent_controller = AffordanceCtrl(**model_param)

    ###
    # folder_path = "/media/lab/NEPTUNE2/field_datasets/row_4/2022-10-14-10-01-08"
    # folder_path = "/media/lab/NEPTUNE2/field_datasets/row_10/2022-10-14-10-41-06"
    folder_path = "/media/lab/NEPTUNE2/field_datasets/row_12/2022-10-28-13-16-37"

    # affordance
    data_pd = pandas.read_csv(os.path.join(folder_path, 'pose.csv'))
    dist_center = data_pd['dist_center'].to_numpy()
    rel_angle = data_pd['rel_angle'].to_numpy()

    # dist_left = dist_center + 3.0
    affordance = np.column_stack([dist_center, rel_angle]).astype(np.float32)

    # image files
    color_file_list = glob.glob(os.path.join(folder_path, 'color', '*.png'))
    color_file_list.sort()


    fig, axes = plt.subplots(2,1)
    handle_gt_dist, = axes[0].plot([], label='ground truth')
    handle_pred_dist, = axes[0].plot([], '--', label='predicted')
    axes[0].set_ylabel('dist_center [m]')
    axes[0].legend()

    handle_gt_heading, = axes[1].plot([])
    handle_pred_heading, = axes[1].plot([], '--')
    axes[1].set_ylabel('rel_angle [rad]')
    plt.pause(1e-3)

    counter = 0
    freq = 15
    for color_file, afford in zip(color_file_list, affordance):
        tic = time.perf_counter()
        raw_img = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)

        affordance_pred = agent_controller.predict_affordance(raw_img)

        handle_gt_dist.set_xdata(np.append(handle_gt_dist.get_xdata(), counter))
        handle_gt_dist.set_ydata(np.append(handle_gt_dist.get_ydata(), afford[0]))
        handle_pred_dist.set_xdata(np.append(handle_pred_dist.get_xdata(), counter))
        handle_pred_dist.set_ydata(np.append(handle_pred_dist.get_ydata(), affordance_pred['dist_center_width'] * 6.0))

        handle_gt_heading.set_xdata(np.append(handle_gt_heading.get_xdata(), counter))
        handle_gt_heading.set_ydata(np.append(handle_gt_heading.get_ydata(), afford[1]))
        handle_pred_heading.set_xdata(np.append(handle_pred_heading.get_xdata(), counter))
        handle_pred_heading.set_ydata(np.append(handle_pred_heading.get_ydata(), affordance_pred['rel_angle']))

        counter += 1
        cv2.imshow('display', raw_img)
        cv2.waitKey(1)

        # 
        axes[0].relim()
        axes[0].autoscale()
        axes[1].relim()
        axes[1].autoscale()
        plt.pause(1e-3)
   
        elapsed_time = time.perf_counter() - tic
        if elapsed_time < 1./ freq:
            time.sleep(1./ freq - elapsed_time)