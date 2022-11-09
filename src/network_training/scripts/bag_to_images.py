#!/usr/bin/env python

"""
Example to use:

./bag_to_images.py /media/lab/NEPTUNE2/slam_data/2022-06-30/drone/trail2_xavier_data_1.bag  ~/tmp  /d435i/color/image_raw/compressed 

"""

import os
import argparse

import numpy as np
import cv2

import rosbag
# from sensor_msgs.msg import CompressedImage

def main():
    """
    Extract a folder of images from a rosbag
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    print("Extract images from %s on topic %s into %s" 
            % (args.bag_file, args.image_topic, args.output_dir)
    )

    bag = rosbag.Bag(args.bag_file, "r")
    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        np_arr = np.fromstring(msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        cv2.imwrite(os.path.join(args.output_dir, "%07i.png" % count), image_np)
        print("Wrote image %i" % count)
        count += 1

    bag.close()

    return

if __name__ == "__main__":
    main()