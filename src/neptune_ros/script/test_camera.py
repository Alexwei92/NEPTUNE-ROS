#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

from sensor_msgs.msg import Image, CompressedImage

class TestCamera(object):
    def __init__(self):
        rospy.init_node("test_camera")
        self.rate = rospy.Rate(10)

        self.color_img = None
        self.depth_img = None
        self.compress_depth_img = None
        self.bridge = CvBridge()

        self.fig, self.axes = plt.subplots(3,1)

        rospy.Subscriber(
            "/d435i/color/image_raw/compressed",
            CompressedImage,
            self.color_image_callback,
            queue_size=10,
        )

        rospy.Subscriber(
            "/d435i/aligned_depth_to_color/image_raw",
            Image,
            self.depth_image_callback,
            queue_size=10,
        )

        rospy.Subscriber(
            "/d435i/aligned_depth_to_color/image_raw/compressed",
            CompressedImage,
            self.compressed_depth_image_callback,
            queue_size=10,
        )

    def color_image_callback(self, msg):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.color_img = cv_img

    def depth_image_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb16')
        self.depth_img = cv_img

    def compressed_depth_image_callback(self, msg):
        # cv_img = np.frombuffer(msg.data, np.uint16)
        # cv_img = cv2.imdecode(cv_img, cv2.IMREAD_COLOR)
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb16')
        self.compress_depth_img = cv_img

    def run(self):
        while not rospy.is_shutdown():
            if self.color_img is not None:
                self.axes[0].imshow(self.color_img)

            if self.depth_img is not None:
                self.axes[1].imshow(self.depth_img)

            if self.compress_depth_img is not None:
                self.axes[2].imshow(self.compress_depth_img)

            plt.pause(1e-3)

            self.rate.sleep()


if __name__ == "__main__":

    handle = TestCamera()
    rospy.sleep(1.0)

    handle.run()
