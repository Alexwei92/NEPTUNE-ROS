#!/usr/bin/env python
import rospy

from nav_msgs.msg import Odometry

class ThrottleOdom(object):
    def __init__(self, input_topic, output_topic):

        self.odom_throttle = Odometry()
        rospy.Subscriber(input_topic, Odometry, self.odom_callback)
        self.pub = rospy.Publisher(output_topic, Odometry, queue_size=10)

    def odom_callback(self, msg):
        odom_throttle = msg
        odom_throttle.header.frame_id = "camera_odom_frame"
        odom_throttle.child_frame_id = "camera_pose_frame"
        self.odom_throttle = odom_throttle

    def publish_odom_throttle(self):
        self.pub.publish(self.odom_throttle)

if __name__ == "__main__":
    rospy.init_node("throttle_t265_odom")

    input_topic = rospy.get_param("~input_topic", "/t265/odom/sample")
    output_topic = rospy.get_param("~output_topic", "/camera/odom/sample_throttled")
    new_rate = rospy.get_param("~new_rate", 25)

    rate = rospy.Rate(new_rate)

    throttle = ThrottleOdom(input_topic, output_topic)
    
    while not rospy.is_shutdown():
        throttle.publish_odom_throttle()
        rate.sleep()
