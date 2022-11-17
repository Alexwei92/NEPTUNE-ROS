"""
Display data coverage and distribution
"""
import os
import pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt

from utils.plot_utils import plot_single_data


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
    row_index = 18
    dataset_dir = "/media/lab/NEPTUNE2/field_datasets/row_" + str(row_index)
    fig, axis = plt.subplots(3,1)
    for i in range(len(axis)):
        plot_single_data(axis[i], data['row_data'][row_index])

    for folder in os.listdir(dataset_dir):
        subfolder = os.path.join(dataset_dir, folder)
        data_pd = pandas.read_csv(os.path.join(subfolder, 'pose.csv'))
        dist_center = data_pd['dist_center'].to_numpy()
        rel_angle = data_pd['rel_angle'].to_numpy()
        pos_x = data_pd['pos_x'].to_numpy()
        pos_y = data_pd['pos_y'].to_numpy()
        heading = data_pd['heading'].to_numpy()

        axis[0].scatter(pos_x, pos_y, s=50, marker='s', alpha=0.1, color='r', edgecolors='none')
        if (heading[0] > np.pi/2) and (heading[0] < 3*np.pi/2):
            idx = 2
        else:
            idx = 1 
        axis[idx].quiver(pos_x, pos_y, np.cos(-heading), np.sin(-heading), color='b', alpha=0.3, scale=50, width=0.001)

    plt.show()
