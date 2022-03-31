#!/usr/bin/env python

""" 
Extract images and telemetry data from a ROS bag.
"""

import os
import argparse
import rosbag
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import shutil, csv, string
import __future__


# define the topics in the bag file
# TODO: put the topic definition into a global file so it can be shared among different programs
image_topics_dict = {
    # prefix    : [topic_name, desired_encoding]
    'color'     : ['/d435i/color/image_raw', 'bgr8'],  # not rgb8
    'depth'     : ['/d435i/aligned_depth_to_color/image_raw', 'mono16'] # 8UC1?
    }

telemetry_topics_dict = {
    # prefix    : [topic_name]
    'cmd'         : '/my_telemetry/rc/in',
    'vel_body'    : '/my_telemetry/velocity_body',
    'gps'         : '/my_telemetry/global_location',
    'local_pose'  : '/my_telemetry/local_pose'
    }


# Write image data
def write_image(bagfile, path):
    bridge = CvBridge()

    for name in image_topics_dict.keys():
        # counter
        count_good, count_bad = 0, 0

        # create folder to put the images in
        path_folder = path+'/'+name
        if not os.path.isdir(path_folder):
            os.makedirs(path_folder)

        # file handler
        f = open(path+'/'+name+'_info.csv','w')
        filewriter = csv.writer(f, delimiter = ',')
        filewriter.writerow(['rosbagTimestamp', 'flag']) # flag=1 means good data, flag=0 means bad data

        for topic, msg, t in bagfile.read_messages(topics=image_topics_dict[name][0]):
            values = [str(t)]
            try:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding=image_topics_dict[name][1])
                cv2.imwrite(os.path.join(path_folder, "frame%06i.png" % count_good), cv_img)
                count_good += 1
                values.append(1)
            except:
                count_bad += 1
                values.append(0)
                pass
            finally:
                filewriter.writerow(values)

        print("Wrote %i %s images." % (count_good, name))
        if count_bad > 0:
            print("[Warning: Found %i broken %s images.]" % (count_bad, name))

        f.close()

    return


# Write telemetry data
def write_telemetry(bagfile, path):
    for topic_name in telemetry_topics_dict.values():
        count = 0
        filename = path+'/'+string.replace(topic_name, '/', '_')+'.csv'
        with open(filename, 'w+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter = ',')
            firstIteration = True #allows header row
            for subtopic, msg, t in bagfile.read_messages(topics=topic_name):
                #parse data from this instant, which is of the form of multiple lines of "Name: value\n"
				#	- put it in the form of a list of 2-element lists
                msgString = str(msg)
                msgList = string.split(msgString, '\n')
                instantaneousListOfData = []
                for nameValuePair in msgList:
                    splitPair = string.split(nameValuePair, ':')
                    for i in range(len(splitPair)):	#should be 0 to 1
                        splitPair[i] = string.strip(splitPair[i])
                    instantaneousListOfData.append(splitPair)
                #write the first row from the first element of each pair
                if firstIteration:	# header
                    headers = ["rosbagTimestamp"]	#first column header
                    for pair in instantaneousListOfData:
                        headers.append(pair[0])
                    filewriter.writerow(headers)
                    firstIteration = False
                # write the value from each pair to the file
                values = [str(t)]	#first column will have rosbag timestamp
                for pair in instantaneousListOfData:
                    if len(pair) > 1:
                        values.append(pair[1])
                filewriter.writerow(values)
                count += 1
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract images and telemetry data from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    #parser.add_argument("output_dir", help="Output directory.")
    #parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    try:
        bag = rosbag.Bag(args.bag_file, "r")
        print("Extract images from '%s'" % (args.bag_file))
    except IOError:
        print("[Error: No such file: '%s']\nFailed!" % (args.bag_file))
        exit(1)

    # Create the output directory
    path = os.path.join('./', args.bag_file[:-4])
    if not os.path.isdir(path):
        os.makedirs(path)
    shutil.copyfile(args.bag_file, path+'/'+bag.filename)

    # Write data
    try:
        write_image(bag, path)
        write_telemetry(bag, path)
        print("Successful.")
    finally:
        bag.close()
    
