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
        'afford_model_path': '/media/lab/NEPTUNE2/field_outputs/row_4/affordance/affordance_model.pt'
    }
    
    agent_controller = AffordanceCtrl(**model_param)

    ###
    folder_path = "/media/lab/NEPTUNE2/field_datasets/row_4/2022-10-14-09-53-49"

    # affordance
    data_pd = pandas.read_csv(os.path.join(folder_path, 'pose.csv'))
    dist_center = data_pd['dist_center'].to_numpy()
    rel_angle = data_pd['rel_angle'].to_numpy()

    dist_left = dist_center + 3.0
    affordance = np.column_stack([dist_center, rel_angle, dist_left]).astype(np.float32)

    # image files
    color_file_list = glob.glob(os.path.join(folder_path, 'color', '*.png'))
    color_file_list.sort()

    dist_center_width_pred = []
    rel_angle_pred = []
    dist_left_width_pred = []
    for color_file, afford in zip(color_file_list, affordance):
        raw_img = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)

        affordance_pred = agent_controller.predict_affordance(raw_img)

        dist_center_width_pred.append(affordance_pred['dist_center_width'])
        rel_angle_pred.append(affordance_pred['rel_angle'])
        dist_left_width_pred.append(affordance_pred['dist_left_width'])

    dist_center_pred = np.array(dist_center_width_pred) * 6.0
    rel_angle_pred = np.array(rel_angle_pred)
    dist_left_pred = np.array(dist_left_width_pred) * 6.0


    fig, axes = plt.subplots(3,1)

    axes[0].plot(affordance[:,0] / 6.0)
    axes[0].plot(dist_center_pred / 6.0)

    axes[1].plot(affordance[:,1] / (np.pi/2))
    axes[1].plot(rel_angle_pred / (np.pi/2))

    axes[2].plot(affordance[:,2] / 6.0 - 0.5)
    axes[2].plot(dist_left_pred / 6.0 - 0.5)

    plt.show()
