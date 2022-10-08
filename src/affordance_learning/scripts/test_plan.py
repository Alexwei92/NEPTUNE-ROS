import os
import pickle
import matplotlib.pyplot as plt

from utils.plot_utils import FieldMapPlot
from utils.navigation_utils import *


if __name__ == "__main__":
    
    curr_dir = os.path.dirname(__file__)
    data_path = os.path.join(curr_dir, "ground_truth/plant_field.pkl")

    with open(data_path, 'rb') as file:
        data = pickle.load(file)
        print("Load data successfully!")

    processed_data = {
        'row': data['row_data'],
        'column': data['column_data'],
    }
    utm_T_local=data['utm_T_local']
    field_bound = {
        'latlon': data['field_bound_latlon'],
        'local': data['field_bound_local'],
    }

    map_handler = FieldMapPlot(
        processed_data,
        utm_T_local,
        field_bound,
        frame='local',
    )


    map_handler.draw_all_rows()
    plt.show()
    