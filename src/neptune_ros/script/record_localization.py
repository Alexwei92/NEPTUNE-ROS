#!/usr/bin/env python

import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped 

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import transformation

def check_matrix_equal(matrix1, matrix2):
    row, col = matrix1.shape
    for i in range(row):
        for j in range(col):
            if matrix1[i, j] != matrix2[i, j]:
                return False
    
    return True

class RecordLocalization(object):
    def __init__(self, rate):

        self.all_state = []
        self.all_transform = []
        self.latest_transform = None
        self.rate = rospy.Rate(rate)

        self.tf_listener = tf.TransformListener()
        rospy.Subscriber("/t265/odom/sample", Odometry, self.odom_callback)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        q_x = msg.pose.pose.orientation.x
        q_y = msg.pose.pose.orientation.y
        q_z = msg.pose.pose.orientation.z
        q_w = msg.pose.pose.orientation.w
        roll, pitch, yaw = tf.transformations.euler_from_quaternion([q_w, q_x, q_y, q_z])

        # current_state = transformation.states2SE3(
        #     [x, y, z, roll, pitch, yaw]
        # )
        current_state = [x, y, z, roll, pitch, yaw]
        self.all_state.append(current_state)

    def get_transform(self, target_frame, source_frame):
        success, transform = self._lookup_transformation(
            self.tf_listener, target_frame, source_frame
        )

        return success, transform

    def _lookup_transformation(self, listener, target_frame, source_frame):
        success = False
        transform = None
        # target_T_source = None
        try:
            transform = listener.lookupTransform(
                target_frame, source_frame, rospy.Time(0)
            )
            # (trans, rot) = listener.lookupTransform(
            #     target_frame, source_frame, rospy.Time(0)
            # )
            # rot_matrix = tf.transformations.quaternion_matrix(
            #     [rot[0], rot[1], rot[2], rot[3]]
            # )
            # rot_matrix[0, 3] = trans[0]
            # rot_matrix[1, 3] = trans[1]
            # rot_matrix[2, 3] = trans[2]
            # target_T_source = rot_matrix.copy()
            success = True

        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            pass
    
        return success, transform

    def run(self):
        try:
            while not rospy.is_shutdown():
                success, transform = self.get_transform('map_rtab', 't265_odom_frame')
                if success:
                    if self.latest_transform is None or (self.latest_transform != transform):
                        self.latest_transform = transform
                        self.all_transform.append(transform)
                        rospy.loginfo("Loop closure detected. Transformation updated!")
                self.rate.sleep()

        except KeyboardInterrupt:
            print("The latest transformation matrix is:")
            print("translationanl: ", self.latest_transform[0])
            print("rotational: ", self.latest_transform[1])
            # TODO: save to file


            all_state = np.array(self.all_state)

            (trans, rot) = self.latest_transform

            rot_matrix = tf.transformations.quaternion_matrix(
                [rot[0], rot[1], rot[2], rot[3]]
            )
            rot_matrix[0, 3] = trans[0]
            rot_matrix[1, 3] = trans[1]
            rot_matrix[2, 3] = trans[2]

            map_T_odom = rot_matrix

            pos = []
            for state in all_state:
                state_se3 = transformation.states2SE3(state)
                state_in_map = map_T_odom.dot(state_se3)
                pos.append(state_in_map[0:3, 3])

            pos = np.array(pos)

            fig = plt.figure()
            # ax = plt.axes(projection ='3d')
            ax = plt.axes()
            # ax.plot3D(pos[:,0], pos[:,1], pos[:,2])
            ax.plot(pos[:,0], pos[:,1])
            ax.set_aspect('equal')

            plt.show()


if __name__ == "__main__":
    rospy.init_node("record_localization")

    recorder = RecordLocalization(200)
    recorder.run()