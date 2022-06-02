#!/usr/bin/env python
import rospy
import tf
import tf2_ros

class DuplicateTFTransform(object):
    def __init__(self, 
        target_frame="camera_odom_frame", 
        source_frame="camera_pose_frame",
        ):

        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()

        self.target_frame = target_frame
        self.source_frame = source_frame

    def broadcast_tf(self, transform):
        self.tf_broadcaster.sendTransform(
            transform[0], # translation
            transform[1], # rotation
            rospy.Time.now(),
            self.source_frame,
            self.target_frame,
        )

    def get_transform(self, target_frame, source_frame):
        success = False
        while not rospy.is_shutdown() and not success:
            success, transform = self._lookup_transformation(
                self.tf_listener, target_frame, source_frame
            )
            if not success:
                rospy.logwarn(
                    "lookupTransform from %s to %s failed"
                    % (target_frame, source_frame)
                )
            rospy.sleep(1 / 200)

        return transform

    def _lookup_transformation(self, listener, target_frame, source_frame):
        success = False
        transform = None
        try:
            transform = listener.lookupTransform(
                target_frame, source_frame, rospy.Time(0)
            )
            success = True
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            pass
    
        return success, transform


if __name__ == "__main__":
    rospy.init_node("duplicate_tf_transform")
    
    target_frame = rospy.get_param("~target_frame", 't265_odom_frame')
    source_frame = rospy.get_param("~source_frame", 't265_pose_frame')
    new_target_frame = rospy.get_param("~new_target_frame", 'camera_odom_frame')
    new_source_frame = rospy.get_param("~new_source_frame", 'camera_pose_frame')

    rate = rospy.Rate(rospy.get_param("~rate", 200))

    tf_duplicater = DuplicateTFTransform(new_target_frame, new_source_frame)

    while not rospy.is_shutdown():
        transform = tf_duplicater.get_transform(target_frame, source_frame)
        tf_duplicater.broadcast_tf(transform)
        rate.sleep()