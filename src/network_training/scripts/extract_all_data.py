#!/usr/bin/env python
"""
Extract from raw data and save them to a folder
"""
import os
import shutil
import numpy as np
import pandas
import pickle
import cv2
from cv_bridge import CvBridge

from tqdm import tqdm

import rosbag
from utils import rosbag_utils
from utils.navigation_utils import *


class ExtractData():
    def __init__(self, 
        bag_path,
        output_folder,
        bag_folder_name,
        field_data,
    ):
        # output config
        self.output_folder = output_folder
        self.bag_folder_name = bag_folder_name

        # field data
        self.field_data = field_data
        self.utm_T_local = field_data['utm_T_local']

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        print(bag_path)
        bag = rosbag.Bag(bag_path, "r")
        self.extract_data_from_bag(bag)

    def extract_data_from_bag(self, bag):
        ## read topics
        color_image = rosbag_utils.get_topic_from_bag(bag, "/d435i/color/image_raw/compressed", False)
        # depth_image = rosbag_utils.get_topic_from_bag(bag, "/d435i/aligned_depth_to_color/image_raw/compressed", False)
        compass_hdg = rosbag_utils.get_topic_from_bag(bag, "/mavros/global_position/compass_hdg", False)
        px4_global_position = rosbag_utils.get_topic_from_bag(bag, "/mavros/global_position/global", False)
        piksi_global_position = rosbag_utils.get_topic_from_bag(bag, "/piksi/navsatfix_best_fix", False)

        ros_time, sync_topics = rosbag_utils.timesync_topics([
            color_image,
            piksi_global_position,
            compass_hdg,
            px4_global_position
        ], printout=False)
    
        color_image_sync = sync_topics[0]
        piksi_global_position_sync = sync_topics[1]
        compass_hdg_sync = sync_topics[2]
        px4_global_position_sync = sync_topics[3]

        # heading
        compass_heading = np.radians(compass_hdg_sync['data'])

        # local position
        local_pos_x = []
        local_pos_y = []
        for i in range(len(piksi_global_position_sync)):
            pos_xy = get_local_xy_from_latlon(
                piksi_global_position_sync['latitude'].iloc[i],
                piksi_global_position_sync['longitude'].iloc[i],
                # px4_global_position_sync['latitude'].iloc[i],
                # px4_global_position_sync['longitude'].iloc[i],
                self.utm_T_local,
            )
            local_pos_x.append(pos_xy[0])
            local_pos_y.append(pos_xy[1])

        local_pos_x = np.array(local_pos_x)
        local_pos_y = np.array(local_pos_y)

        results = {
            'time': piksi_global_position_sync['ros_time'],
            # 'time': px4_global_position_sync['ros_time'],
            'compass_heading': compass_heading,
            'local_pos_x': local_pos_x,
            'local_pos_y': local_pos_y,
        }
        results = pandas.DataFrame(results)

        ## bitmask1: check in polygon
        all_index = []
        for x, y in zip(local_pos_x, local_pos_y):
            current_pose = [x, y]

            index = find_area_index(current_pose, self.field_data['row_data'])
            if index is not None:
                all_index.append(index)

            if len(all_index) > 10:
                break

        row_index = int(np.array(all_index).mean())
        print("Flying in row %d" % row_index)

        current_polygon = Polygon([vertex for vertex in self.field_data['row_data'][row_index]['vertice_actual']])
        bitmask1 = []
        for x, y in zip(local_pos_x, local_pos_y):
            current_pose = [x, y]
            bitmask1.append(whether_in_polygon(current_pose, current_polygon))

        ## bitmask2: check heading
        if local_pos_x[0] > self.field_data['row_data'][row_index]['treelines_actual'][0][1][0]:
            direction = -1 # from east to west
            compass_heading_range = [np.pi, 2*np.pi]
        else:
            direction = 1 # from west to east
            compass_heading_range = [0, np.pi]
        print("Flying direction: %d" % direction)

        bitmask2 = []
        for i in range(len(compass_heading)):
            res = (compass_heading_range[0] <= compass_heading.iloc[i] <= compass_heading_range[1])
            bitmask2.append(res)

        ## Apply bitmasks
        filtered_results = pandas.DataFrame()
        for i in range(len(results)):
            if bitmask1[i] and bitmask2[i]:
                res = {}
                res['time'] = results['time'].iloc[i]
                res['heading'] = wrap_2pi(results['compass_heading'].iloc[i] - np.pi/2)
                res['pos_x'] = results['local_pos_x'].iloc[i]
                res['pos_y'] = results['local_pos_y'].iloc[i]
                res['index'] = i
                filtered_results = filtered_results.append(res, ignore_index=True)

        print("{:.2%} of valid data".format(float(len(filtered_results)) / len(results)))

        ## read images
        bridge = CvBridge()

        color_image_list = []
        # depth_image_list = []
        for i in range(len(filtered_results)):
            # rgb
            idx = int(filtered_results['index'].iloc[i])
            np_arr = np.frombuffer(color_image_sync['data'].iloc[idx], np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # RGB
            color_image_list.append(image_np)

            # depth
            # np_arr = np.frombuffer(depth_image_sync['data'].iloc[idx], np.uint8)
            # image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # depth_image_list.append(image_np)

        print("%d images extracted" % len(color_image_list))

        ### calculate affordance
        current_row_data = self.field_data['row_data'][row_index]
        reference_heading = np.arctan2(
            current_row_data['centerline_actual'][1][1] - current_row_data['centerline_actual'][0][1],
            current_row_data['centerline_actual'][1][0] - current_row_data['centerline_actual'][0][0],
        )
        if direction < 0:
            reference_line = np.flip(current_row_data['centerline_actual'], axis=0)
            reference_heading += np.pi
        else:
            reference_line = current_row_data['centerline_actual']
        reference_heading = wrap_2pi(reference_heading)

        rel_angle = []
        dist_center = []
        for i in range(len(filtered_results)):
            point = [
                filtered_results['pos_x'].iloc[i],
                filtered_results['pos_y'].iloc[i],
            ]
            heading = filtered_results['heading'].iloc[i]
            lat_proj, lon_proj = get_projection_point2line(point, reference_line)

            rel_angle.append(wrap_pi(heading - reference_heading))
            dist_center.append(lat_proj)

        filtered_results['dist_center'] = dist_center
        filtered_results['rel_angle'] = rel_angle

        ## save data
        row_folder_name = 'row_' + str(row_index)
        row_folder = os.path.join(self.output_folder, row_folder_name)

        if not os.path.isdir(row_folder):
            os.mkdir(row_folder)

        output_data_folder = os.path.join(row_folder, self.bag_folder_name)
        if os.path.isdir(output_data_folder):
            shutil.rmtree(output_data_folder)

        os.makedirs(output_data_folder)
        os.makedirs(output_data_folder + '/color')

        filtered_results.to_csv(
            os.path.join(output_data_folder, 'pose.csv'),
            columns=['time', 'pos_x', 'pos_y', 'heading', 'dist_center', 'rel_angle'],
            index=False,
        )

        for i in range(len(color_image_list)): 
            cv2.imwrite(os.path.join(output_data_folder, "color/%07i.png" % i), color_image_list[i])


if __name__ == "__main__":
    root_folder_path = '/media/lab/NEPTUNE2/field_raw_datasets/2022-11-15'
    output_folder = '/media/lab/NEPTUNE2/field_datasets'

    #### 
    curr_dir = os.path.dirname(__file__)
    ground_truth_path = os.path.join(curr_dir, "ground_truth/plant_field.pkl")

    # Configure map
    with open(ground_truth_path, 'rb') as file:
        field_data = pickle.load(file)

    for file in tqdm(os.listdir(root_folder_path)):
        bag_folder_name = file[11:-4]
        bag_path = os.path.join(root_folder_path, file)
        ExtractData(
            bag_path,
            output_folder,
            bag_folder_name,
            field_data,
        )