#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image

class DepthControl():
    def __init__(self):
        rospy.init_node("depth_control")
        self.rate = rospy.Rate(15)

        self.init_variable()
        self.define_subscriber()

    def init_variable(self):
        # flag
        self.camera_is_ready = False

        # image
        self.bridge = CvBridge()
        self.depth_img = None

        # timestep
        self.last_depth_timestamp = None
        self.camera_timeout_counter = 0

    def define_subscriber(self):
        # camera depth image
        rospy.Subscriber(
            "/d435i/aligned_depth_to_color/image_raw",
            Image,
            self.depth_image_callback,
            queue_size=5,
            tcp_nodelay=True
        )

    def depth_image_callback(self, msg):
        if msg is not None:
            self.camera_is_ready = True
            self.last_depth_timestamp = msg.header.stamp
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
            self.depth_img = cv_img
            # self.depth_img = np.array(cv_img / 65535, dtype=float)

        print(self.depth_img.max(), self.depth_img.min())

    def check_status(self):
        # check topic timeout
        if self.camera_is_ready and (rospy.Time.now() - self.last_depth_timestamp).to_sec() > 0.5:
            rospy.logwarn_throttle(1, 'Color image stream rate is slow!')
            self.camera_timeout_counter += 1

        # reset if timeout counter exceed maximum value
        if self.camera_timeout_counter > (15 * 2):
            rospy.logerr("Connection to camera lost!")
            self.reset()
        
    def reset(self):
        self.init_variable()
        rospy.loginfo('Reset!')

    def run(self):
        while not rospy.is_shutdown():
            # check status
            self.check_status()

            if self.depth_img is not None:
                cv2.imshow("depth", self.depth_img)

            key = cv2.waitKey(1) & 0xFF
            if (key == 27 or key == ord('q')):
                break

            self.rate.sleep()

if __name__ == "__main__":
    handle = DepthControl()
    handle.run()
    
