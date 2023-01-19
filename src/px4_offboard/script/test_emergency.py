import rospy
from mavros_msgs.msg import EstimatorStatus

if __name__ == "__main__":
    rospy.init_node("agent_control")

    pub = rospy.Publisher(
        "/mavros/estimator_status",
        EstimatorStatus,
        queue_size=1,
    )

    msg = EstimatorStatus()
    msg.header.stamp = rospy.Time.now()
    msg.pos_horiz_abs_status_flag = False

    pub.publish(msg)
