import os
import rospy
import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from mavros_msgs.msg import RCIn

from controller import VAECtrl

from utils.math_utils import euler_from_quaternion

######################
curr_dir = os.path.dirname(os.path.abspath(__file__))
trained_model_dir = os.path.abspath(os.path.join(curr_dir, "../../../model/vae"))

model_config = {
    'vae_model_path': os.path.join(trained_model_dir, 'vanilla_vae_model_z_1000.pt'),
    'latent_model_path': os.path.join(trained_model_dir, 'latent_ctrl_vanilla_vae_model_z_1000.pt'),
}
######################

class AgentController():
    def __init__(self):
        rospy.init_node("agent_control")
        self.rate = rospy.Rate(30)

        self.init_agent()
        
        self.init_variable()
        self.define_subscriber()
        rospy.loginfo("The agent controller has been initialized!")

    def init_agent(self):
        self.agent = VAECtrl(**model_config)

    def init_variable(self):
        # image
        self.color_img = None

        # mavros states
        self.roll = 0.0
        self.pitch = 0.0
        self.body_linear_x = 0.0
        self.body_linear_y = 0.0
        self.body_angular_z = 0.0

        # timestamp
        self.last_color_timestamp = None
        self.last_local_position_timestamp = None 

    def define_subscriber(self):
        # camera color image
        rospy.Subscriber(
            "/d435i/color/image_raw/",
            Image,
            self.color_img_callback,
            queue_size=5,
        )

        # local odom
        rospy.Subscriber(
            "/mavros/local_position/odom",
            Odometry,
            self.local_position_odom_callback,
            queue_size=5,
        )

        # velocity body
        rospy.Subscriber(
            "/mavros/local_position/velocity_body",
            TwistStamped,
            self.velocity_body_callback,
            queue_size=5,
        )

        # rc in
        rospy.Subscriber(
            "/mavros/rc/in",
            RCIn,
            self.rc_in_callback,
            queue_size=5,
        )

    def color_img_callback(self, msg):
        if msg is not None:
            self.last_color_timestamp = msg.header.stamp
            self.color_img = msg.data

    def local_position_odom_callback(self, msg):
        if msg is not None:
            self.last_local_position_timestamp = msg.header.stamp
            self.roll, self.pitch, _ = euler_from_quaternion(msg.pose.pose.orientation)

    def velocity_body_callback(self, msg):
        if msg is not None:
            self.body_linear_x = msg.twist.linear.x
            self.body_linear_y = msg.twist.linear.y
            self.body_angular_z = msg.twist.angular.z

    def rc_in_callback(self, msg):
        return

    def run(self):
        while not rospy.is_shutdown():
            # check topic timeout
            if (self.last_color_timestamp is None) or (
                rospy.Time.now() - self.last_color_timestamp).to_sec() > 0.1:
                rospy.logwarn('Color image stream is lost!')

            if (self.last_local_position_timestamp is None) or (
                rospy.Time.now() - self.last_color_timestamp).to_sec() > 0.1:
                rospy.logwarn('Mavros connection to pixhawk is lost!')

            # state extra
            state_extra = np.array([
                self.roll,
                self.pitch,
                self.body_linear_x,
                self.body_linear_y,
                self.body_angular_z,
            ], dtype=np.float32)

            # predict action
            agent_output = self.agent.predict(self.color_img, is_bgr=False, state_extra=state_extra)

            self.rate.sleep()


if "__name__" == "__main__":
    handler = AgentController()
    handler.run()
