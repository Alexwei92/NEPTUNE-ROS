#!/usr/bin/env python
import rospy
from mavros_msgs.msg import EstimatorStatus

if __name__ == "__main__":
    rospy.init_node("test_emergency")
    rate = rospy.Rate(1)

    pub = rospy.Publisher(
        "/mavros/estimator_status",
        EstimatorStatus,
        queue_size=1,
    )

    msg = EstimatorStatus()
    msg.header.stamp = rospy.Time.now()
    msg.accel_error_status_flag = True
    msg.velocity_horiz_status_flag = True
    msg.velocity_vert_status_flag = True
    msg.pos_horiz_rel_status_flag = True
    msg.pos_horiz_abs_status_flag = True
    msg.pos_vert_abs_status_flag = True
    msg.pos_vert_agl_status_flag = False
    msg.const_pos_mode_status_flag = False
    msg.pred_pos_horiz_rel_status_flag = True
    msg.pred_pos_horiz_abs_status_flag = True
    msg.gps_glitch_status_flag = False
    msg.accel_error_status_flag = False

    msg.pos_horiz_abs_status_flag = False

    for i in range(5):
        pub.publish(msg)
        rate.sleep()