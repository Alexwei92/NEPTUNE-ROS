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

import rospy
from sensor_msgs.msg import Image, CompressedImage

from utils.plot_utils import FieldMapPlot, plot_single_data
from controller import AffordanceCtrl


BLACK  = (0,0,0)
WHITE  = (255,255,255)
RED    = (0,0,255)
BLUE   = (255,0,0)
GREEN  = (0,255,0)
YELLOW = (0,255,255)
CYAN   = (255,255,0)


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


    print('Load field data successfully!')

    dataset_dir = "/media/lab/NEPTUNE2/field_datasets/row_12"
    fig, axis = plt.subplots(1,1)
    plot_single_data(axis, data['row_data'][12])
    for folder in os.listdir(dataset_dir):
        subfolder = os.path.join(dataset_dir, folder)
        data_pd = pandas.read_csv(os.path.join(subfolder, 'pose.csv'))
        dist_center = data_pd['dist_center'].to_numpy()
        rel_angle = data_pd['rel_angle'].to_numpy()
        pos_x = data_pd['pos_x'].to_numpy()
        pos_y = data_pd['pos_y'].to_numpy()

        axis.scatter(pos_x, pos_y, s=20, alpha=0.1, color='r', edgecolors='none')

    ### Load Network
    model_param = {
        'afford_model_path': '/media/lab/NEPTUNE2/field_outputs/row_4_10/affordance/affordance_model.pt'
    }
    
    agent_controller = AffordanceCtrl(**model_param)

    ###
    # folder_path = "/media/lab/NEPTUNE2/field_datasets/row_4/2022-10-14-10-01-08"
    # # folder_path = "/media/lab/NEPTUNE2/field_datasets/row_10/2022-10-14-10-41-06"
    # # folder_path = "/media/lab/NEPTUNE2/field_datasets/row_12/2022-10-28-13-30-28"

    # # affordance
    # data_pd = pandas.read_csv(os.path.join(folder_path, 'pose.csv'))
    # dist_center = data_pd['dist_center'].to_numpy()
    # rel_angle = data_pd['rel_angle'].to_numpy()
    # pos_x = data_pd['pos_x'].to_numpy()
    # pos_y = data_pd['pos_y'].to_numpy()

    # # dist_left = dist_center + 3.0
    # affordance = np.column_stack([dist_center, rel_angle]).astype(np.float32)

    # # image files
    # color_file_list = glob.glob(os.path.join(folder_path, 'color', '*.png'))
    # color_file_list.sort()

    # dist_center_width_pred = []
    # rel_angle_pred = []
    
    # for index in range(len(color_file_list)):
    #     raw_img_list = []
    #     for j in [0,2,4,10]:
    #         color_file = color_file_list[max(0, index-j)]
    #         raw_img_list.append(cv2.imread(color_file, cv2.IMREAD_UNCHANGED))
 
    #     affordance_pred = agent_controller.predict_affordance(raw_img_list)

    #     dist_center_width_pred.append(affordance_pred['dist_center_width'])
    #     rel_angle_pred.append(affordance_pred['rel_angle'])

    # dist_center_pred = np.array(dist_center_width_pred) * 6.0
    # rel_angle_pred = np.array(rel_angle_pred)


    # fig, axes = plt.subplots(2,1)

    # axes[0].plot(affordance[:,0], label='ground truth')
    # axes[0].plot(dist_center_pred, '--', label='predicted')
    # axes[0].set_ylabel('dist_center [m]')
    # axes[0].legend()

    # axes[1].plot(affordance[:,1])
    # axes[1].plot(rel_angle_pred, '--')
    # axes[1].set_ylabel('rel_angle [rad]')

    plt.show()
