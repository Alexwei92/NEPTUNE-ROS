#!/usr/bin/env python3
import os
import numpy as np
import cv2
import time
from collections import deque
import rospy
from sensor_msgs.msg import Image

from controller import AffordanceCtrl

class CameraStream():
    LOOP_RATE = 15 # Hz
    TIME_OUT  = 0.5 # second

    def __init__(self):
        rospy.init_node("camera_stream")
        self.rate = rospy.Rate(self.LOOP_RATE)
        self.get_ros_parameter()
        self.define_subscriber()
        self.init_variables()
        rospy.loginfo("Node Started!")

    def get_ros_parameter(self):
        self.frame_rate = rospy.get_param("/d435i/realsense2_camera/color_fps", 15)

    def init_variables(self):
        self.has_initialized = False
        
        # image queue
        if self.frame_rate == 6:
            maxlen = 10
        elif self.frame_rate == 15:
            maxlen = 25
        else:
            maxlen = 30

        self.color_img_queue = deque(maxlen=maxlen)
        self.depth_img_queue = deque(maxlen=maxlen)

        self.last_color_timestamp = None
        self.last_depth_timestamp = None

    def define_subscriber(self):
        rospy.Subscriber(
            "/d435i/color/image_raw",
            Image,
            self.color_image_callback,
            queue_size=10,
        )

        rospy.Subscriber(
            "/d435i/aligned_depth_to_color/image_raw",
            Image,
            self.depth_image_callback,
            queue_size=10,
        )

    def color_image_callback(self, msg):
        timestamp = msg.header.stamp
        self.last_color_timestamp = timestamp
        img_np = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        self.color_img_queue.append(img_np)
       
    def depth_image_callback(self, msg):
        timestamp = msg.header.stamp
        self.last_depth_timestamp = timestamp
        img_np = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width, -1).squeeze(2)
        self.depth_img_queue.append(img_np)

    def load_affordance_model(self, **model_param):
        self.agent_controller = AffordanceCtrl(**model_param)
        rospy.loginfo("Load Affordance Model Successfully!")

    def run(self):
        while not rospy.is_shutdown():
            tic = time.perf_counter()
            # check topic timeout
            if (self.last_color_timestamp is None) or (rospy.Time.now() - self.last_color_timestamp).to_sec() > self.TIME_OUT:
                rospy.logwarn('Color image stream is lost!')

            if (self.last_depth_timestamp is None) or (rospy.Time.now() - self.last_depth_timestamp).to_sec() > self.TIME_OUT:
                rospy.logwarn('Depth image stream is lost!')

            # for visualize only
            last_color_img = self.color_img_queue[-1]
            if last_color_img is not None:
                cv2.imshow('color', cv2.cvtColor(last_color_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
    
            if self.frame_rate == 6:
                index_list = [1, 2, 4, 10]
            else:
                index_list = [1, 5, 10, 25]

            color_img_list = []
            for idx in index_list:
                color_img_list.append(self.color_img_queue[-min(idx, len(self.color_img_queue))])
            color_img_list.reverse()

            results = self.agent_controller.predict_affordance(color_img_list)
            # print(results)

            self.rate.sleep()
            print(1. / (time.perf_counter() - tic))

if __name__ == "__main__":
    # init
    handler = CameraStream()

    # load trained model
    curr_dir = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join("../../../", curr_dir))

    model_param = {
        'afford_model_path': os.path.join(root_dir, "model/affordance/affordance_model.pt")
    }
    handler.load_affordance_model(**model_param)

    rospy.sleep(2.0)
    handler.run()