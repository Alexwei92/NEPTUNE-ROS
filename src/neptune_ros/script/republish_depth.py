#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CompressedImage

class RepublishDepth(object):
    def __init__(self):
        rospy.init_node("republish_depth")

        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher(
            "/d435i/aligned_depth_to_color/image_raw",
             Image,
             queue_size=10,
        )

        rospy.Subscriber(
            "/d435i/aligned_depth_to_color/image_raw/compressed",
            CompressedImage,
            self.compressed_depth_image_callback,
            queue_size=10,
        )
        
    def compressed_depth_image_callback(self, msg):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        image_msg = self.bridge.cv2_to_imgmsg(255 - cv_img, encoding='mono8')
        image_msg.header.stamp = msg.header.stamp
        image_msg.header.frame_id = 'd435i_color_optical_frame'
        self.image_pub.publish(image_msg)
   
if __name__ == "__main__":
    handle = RepublishDepth()
    rospy.spin()