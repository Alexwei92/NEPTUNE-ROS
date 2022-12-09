#!/usr/bin/env python
import rospy
import numpy as np

from collections import deque

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from mavros_msgs.msg import (
    RCIn,
    RCOut,
)

def wrap_pi(angle):
    while angle < -np.pi:
        angle += 2.0 * np.pi

    while angle > np.pi:
        angle -= 2.0 * np.pi

    return angle

class ReplayMAVROS(object):
    def __init__(self):
        rospy.init_node("replay_mavros")

        rate = rospy.get_param("~rate", 200)
        self.dt = 1. / rate
        self.rate = rospy.Rate(rate)
        
        self.init_variable()
        self.define_subscriber()
        self.define_publisher()

    def init_variable(self):
        self.motor_angle = np.zeros(4)
        self.path = deque(maxlen=5000)

    def define_subscriber(self):
        # Local Poses
        rospy.Subscriber(
            "/mavros/local_position/pose",
            PoseStamped,
            self.local_pose_callback,
            queue_size=10,
        )

        # # RC In
        # rospy.Subscriber(
        #     "/mavros/rc/in",
        #     RCIn,
        #     self.rc_in_callback,
        #     queue_size=10,
        # )

        # RC Out
        rospy.Subscriber(
            "/mavros/rc/out",
            RCOut,
            self.rc_out_callback,
            queue_size=10,
        )
    
    def rc_out_callback(self, msg):
        for i in range(4):
            angle_diff = (msg.channels[i] - 1050) * 100.0 * self.dt
            self.motor_angle[i] = wrap_pi(self.motor_angle[i] + angle_diff)

    def local_pose_callback(self, msg):
        self.path.append(msg)

    def define_publisher(self):
        self.joint_states_pub = rospy.Publisher(
            "/joint_states",
            JointState,
            queue_size=10,
        )

        self.path_pub = rospy.Publisher(
            "/path",
            Path,
            queue_size=5,
        )

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [
            'rotor1_joint',
            'rotor2_joint',
            'rotor3_joint',
            'rotor4_joint',
        ]
        msg.position = [self.motor_angle[i] for i in range(4)]
        self.joint_states_pub.publish(msg)

    def publish_path(self):
        msg = Path()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.poses = [self.path[i] for i in range(len(self.path))]
        self.path_pub.publish(msg)
    
    def run(self):
        while not rospy.is_shutdown():
            self.publish_joint_states()
            if len(self.path) % 10 == 0:
                self.publish_path()
            self.rate.sleep()

if __name__ == "__main__":
    handle = ReplayMAVROS()
    handle.run()