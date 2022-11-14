"""
Test Affordance Net Performance
"""
import os, glob
import pickle
import pandas
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.plot_utils import plot_single_data
from controller import AffordanceCtrl


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


    # print('Load field data successfully!')

    ### Load Network
    model_param = {
        'afford_model_path': '/media/lab/NEPTUNE2/field_outputs/row_4_10_13/affordance/affordance_model.pt'
    }
    
    agent_controller = AffordanceCtrl(**model_param)

    ###
    # folder_path = "/media/lab/NEPTUNE2/field_datasets/row_4/2022-10-14-10-01-08"
    # folder_path = "/media/lab/NEPTUNE2/field_datasets/row_10/2022-10-14-10-41-06"
    folder_path = "/media/lab/NEPTUNE2/field_datasets/row_12/2022-10-28-13-30-28"
    # folder_path = "/media/lab/NEPTUNE2/field_datasets/row_13/2022-11-06-15-44-34"
    # affordance
    data_pd = pandas.read_csv(os.path.join(folder_path, 'pose.csv'))
    dist_center = data_pd['dist_center'].to_numpy()
    rel_angle = data_pd['rel_angle'].to_numpy()
    pos_x = data_pd['pos_x'].to_numpy()
    pos_y = data_pd['pos_y'].to_numpy()
    heading = data_pd['heading'].to_numpy()

    # dist_left = dist_center + 3.0
    affordance = np.column_stack([dist_center, rel_angle]).astype(np.float32)

    # image files
    color_file_list = glob.glob(os.path.join(folder_path, 'color', '*.png'))
    color_file_list.sort()

    dist_center_width_pred = []
    rel_angle_pred = []
    
    for index in range(len(color_file_list)):
        raw_img_list = []
        for j in [0,2,4,10]:
            color_file = color_file_list[max(0, index-j)]
            img_bgr = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            raw_img_list.append(img_rgb)
 
        affordance_pred = agent_controller.predict_affordance(raw_img_list)

        dist_center_width_pred.append(affordance_pred['dist_center_width'])
        rel_angle_pred.append(affordance_pred['rel_angle'])

    dist_center_pred = np.array(dist_center_width_pred) * 6.0
    rel_angle_pred = np.array(rel_angle_pred)


    fig, axes = plt.subplots(3,1)
    plot_single_data(axes[0], data['row_data'][12])
    axes[0].plot(pos_x, pos_y, color='r', linewidth=1.0)

    axes[1].plot(affordance[:,0], label='ground truth')
    axes[1].plot(dist_center_pred, '--', label='predicted')
    axes[1].set_ylabel('dist_center [m]')
    axes[1].legend()

    axes[2].plot(affordance[:,1])
    axes[2].plot(rel_angle_pred, '--')
    axes[2].set_ylabel('rel_angle [rad]')

    rel_angle_diff = abs(affordance[:,1] - rel_angle_pred)
    index_range = rel_angle_diff > np.radians(15)
    index_array = np.where(index_range==True)[0]
    last_index = 0
    axes[0].scatter(pos_x[index_range], pos_y[index_range], s=5, color='r', facecolor='none')
    for j in index_array:
        if j - last_index > 3:
            axes[0].text(pos_x[int(j)], pos_y[int(j)], s=str(j), fontsize=6)
        last_index = j
    axes[0].quiver(pos_x[index_range], pos_y[index_range], np.cos(-heading[index_range]), np.sin(-heading[index_range]),
        color='r', alpha=0.7, scale=30, width=0.002)


    fig2, axes2 = plt.subplots(2,1)

    axes2[0].plot(dist_center_pred - affordance[:,0])
    axes2[0].set_ylabel('dist_center error [m]')

    axes2[1].plot(rel_angle_pred - affordance[:,1])
    axes2[1].set_ylabel('rel_angle error [rad]')

    plt.show()
