#!/usr/bin/env python
"""
Extract from human demonstration data and save them to a folder
"""
import os
import shutil
import numpy as np
import pickle
import pandas
import cv2
from cv_bridge import CvBridge

from tqdm import tqdm

import rosbag
from utils import rosbag_utils
from utils.navigation_utils import *


class ExtractHumanData():
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
        # depth_image = rosbag_utils.get_topic_from_bag(bag, "/d435i/aligned_depth_to_color/image_raw/compressedDepth", False)
        compass_hdg = rosbag_utils.get_topic_from_bag(bag, "/mavros/global_position/compass_hdg", False)
        global_position = rosbag_utils.get_topic_from_bag(bag, "/mavros/global_position/global", False)
        local_position = rosbag_utils.get_topic_from_bag(bag, "/mavros/local_position/odom", False)
        velocity_body = rosbag_utils.get_topic_from_bag(bag, "/mavros/local_position/velocity_body", False)
        rc_in = rosbag_utils.get_topic_from_bag(bag, "/mavros/rc/in", False)
        setpoint_raw = rosbag_utils.get_topic_from_bag(bag, "/mavros/setpoint_raw/local", False)
        my_pid = rosbag_utils.get_topic_from_bag(bag, "/my_controller/pos_z_pid", False)

        # crop data
        start_offset = 1.0
        stop_offset = 3.0
        offboard_start_time = my_pid['ros_time'].iloc[0] + start_offset
        offboard_stop_time = my_pid['ros_time'].iloc[-1] - stop_offset

        color_image_crop = rosbag_utils.crop_data_with_start_end_time(color_image, offboard_start_time, offboard_stop_time)
        global_position_crop = rosbag_utils.crop_data_with_start_end_time(global_position, offboard_start_time, offboard_stop_time)
        compass_hdg_crop = rosbag_utils.crop_data_with_start_end_time(compass_hdg, offboard_start_time, offboard_stop_time)
        local_position_crop = rosbag_utils.crop_data_with_start_end_time(local_position, offboard_start_time, offboard_stop_time)
        velocity_body_crop = rosbag_utils.crop_data_with_start_end_time(velocity_body, offboard_start_time, offboard_stop_time)
        rc_in_crop = rosbag_utils.crop_data_with_start_end_time(rc_in, offboard_start_time, offboard_stop_time)
        setpoint_raw_crop = rosbag_utils.crop_data_with_start_end_time(setpoint_raw, offboard_start_time, offboard_stop_time)

        # sync data
        ros_time, sync_topics = rosbag_utils.timesync_topics([
            color_image_crop,
            setpoint_raw_crop,
            compass_hdg_crop,
            global_position_crop,
            local_position_crop,
            velocity_body_crop,
        ], printout=False)

        color_image_sync = sync_topics[0]
        setpoint_raw_sync = sync_topics[1]
        compass_hdg_sync = sync_topics[2]
        global_position_sync = sync_topics[3]
        local_position_sync = sync_topics[4]
        velocity_body_sync = sync_topics[5]

        # heading
        compass_heading = np.radians(compass_hdg_sync['data'])
        heading = []
        for i in range(len(compass_heading)):
            heading.append(wrap_2pi(compass_heading.iloc[i] - np.pi/2))
        heading = np.array(heading)

        # local position utm
        utm_local_pos_x = []
        utm_local_pos_y = []
        for i in range(len(global_position_sync)):
            pos_xy = get_local_xy_from_latlon(
                global_position_sync['latitude'].iloc[i],
                global_position_sync['longitude'].iloc[i],
                self.utm_T_local,
            )
            utm_local_pos_x.append(pos_xy[0])
            utm_local_pos_y.append(pos_xy[1])

        utm_local_pos_x = np.array(utm_local_pos_x)
        utm_local_pos_y = np.array(utm_local_pos_y)

        # local position odom
        odom_local_pos_z = []
        for pose_msg in local_position_sync['pose']:
            odom_local_pos_z.append(pose_msg.pose.position.z)
        odom_local_pos_z = np.array(odom_local_pos_z)

        # velocity body
        linear_x = []
        for twist_msg in velocity_body_sync['twist']:
            linear_x.append(twist_msg.linear.x)
        linear_x = np.array(linear_x)

        angular_z = []
        for twist_msg in velocity_body_sync['twist']:
            angular_z.append(twist_msg.angular.z)
        angular_z = np.array(angular_z)

        # human input
        control_cmd = setpoint_raw_sync['yaw_rate'].to_numpy()
        control_cmd = -control_cmd / (45.0 * np.pi / 180)
        control_cmd[abs(control_cmd) < 1e-2] = 0.0

        ## read images
        bridge = CvBridge()

        color_image_list = []
        for i in range(len(color_image_sync)):
            # rgb
            np_arr = np.frombuffer(color_image_sync['data'].iloc[i], np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # RGB
            color_image_list.append(image_np)

        print("%d images extracted!" % len(color_image_list))
        
        ## save data
        states = pandas.DataFrame()
        states['time'] = color_image_sync['ros_time'].to_numpy()
        states['utm_pos_x'] = utm_local_pos_x
        states['utm_pos_y'] = utm_local_pos_y
        states['heading'] = heading
        states['odom_pos_z'] = odom_local_pos_z
        states['yaw_rate'] = angular_z
        states['control_cmd'] = control_cmd

        parent_folder_name = os.path.join(self.output_folder, "human_data")
        if not os.path.isdir(parent_folder_name):
            os.mkdir(parent_folder_name)

        output_data_folder = os.path.join(parent_folder_name, self.bag_folder_name)
        if os.path.isdir(output_data_folder):
            shutil.rmtree(output_data_folder)

        os.makedirs(output_data_folder)
        os.makedirs(output_data_folder + '/color')

        states.to_csv(
            os.path.join(output_data_folder, 'states.csv'),
            columns=['time', 'utm_pos_x', 'utm_pos_y', 'heading', 'odom_pos_z', 'yaw_rate', 'control_cmd'],
            index=False,
        )

        for i in range(len(color_image_list)): 
            cv2.imwrite(os.path.join(output_data_folder, "color/%07i.png" % i), color_image_list[i])


if __name__ == "__main__":
    root_folder_path = '/media/lab/NEPTUNE2/field_raw_datasets/2022-12-06_Human'
    output_folder = '/media/lab/NEPTUNE2/field_datasets'

    #### 
    curr_dir = os.path.dirname(__file__)
    ground_truth_path = os.path.join(curr_dir, "ground_truth/plant_field.pkl")

    # Configure map
    with open(ground_truth_path, 'rb') as file:
        field_data = pickle.load(file)

    for file in tqdm(os.listdir(root_folder_path)):
        bag_folder_name = file[4:-4]
        bag_path = os.path.join(root_folder_path, file)
        ExtractHumanData(
            bag_path,
            output_folder,
            bag_folder_name,
            field_data,
        )
