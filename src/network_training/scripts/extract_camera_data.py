#!/usr/bin/env python
"""
Extract from raw camera data and save them to a folder
"""
import os
import shutil
import numpy as np
import cv2
from cv_bridge import CvBridge

from tqdm import tqdm

import rosbag
from utils import rosbag_utils

class ExtractCameraData():
    def __init__(self, 
        bag_path,
        output_folder,
        bag_folder_name,
    ):
        # output config
        self.output_folder = output_folder
        self.bag_folder_name = bag_folder_name

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        print(bag_path)
        bag = rosbag.Bag(bag_path, "r")
        self.extract_data_from_bag(bag)

    def extract_data_from_bag(self, bag):
        ## read topics
        color_image = rosbag_utils.get_topic_from_bag(bag, "/d435i/color/image_raw/compressed", False)
        # depth_image = rosbag_utils.get_topic_from_bag(bag, "/d435i/aligned_depth_to_color/image_raw", False)

        ## read images
        bridge = CvBridge()

        color_image_list = []
        # depth_image_list = []
        for i in range(len(color_image)):
            # rgb
            np_arr = np.frombuffer(color_image['data'].iloc[i], np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # RGB
            color_image_list.append(image_np)

            # depth
            # np_arr = np.frombuffer(depth_image['data'].iloc[i], np.uint8)
            # image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # depth_image_list.append(image_np)

        print("%d images extracted" % len(color_image_list))

        ## save data
        output_data_folder = os.path.join(self.output_folder, self.bag_folder_name)
        if os.path.isdir(output_data_folder):
            shutil.rmtree(output_data_folder)

        os.makedirs(output_data_folder)
        os.makedirs(output_data_folder + '/color')

        for i in range(len(color_image_list)): 
            cv2.imwrite(os.path.join(output_data_folder, "color/%07i.png" % i), color_image_list[i])


if __name__ == "__main__":
    root_folder_path = '/media/lab/NEPTUNE2/field_raw_datasets/2022-12-13_Camera_Only'
    output_folder = '/media/lab/NEPTUNE2/field_datasets/vae_extra'

    for file in tqdm(os.listdir(root_folder_path)):
        bag_folder_name = file[7:-4]
        bag_path = os.path.join(root_folder_path, file)
        ExtractCameraData(
            bag_path,
            output_folder,
            bag_folder_name,
        )