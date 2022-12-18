import os
import rospy
import time
import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from mavros_msgs.msg import RCIn
from px4_offboard.msg import ControlCmd

from controller import VAECtrl
from utils.math_utils import euler_from_quaternion, constrain_value

######################
curr_dir = os.path.dirname(os.path.abspath(__file__))
trained_model_dir = os.path.abspath(os.path.join(curr_dir, "../../../model/vae"))

model_config = {
    'vae_model_path': os.path.join(trained_model_dir, 'vanilla_vae_model_z_1000.pt'),
    'latent_model_path': os.path.join(trained_model_dir, 'latent_ctrl_vanilla_vae_model_z_1000.pt'),
}
######################

class AgentControl():
    def __init__(self):
        rospy.init_node("agent_control")
        self.rate = rospy.Rate(15)

        self.start_time = time.time()

        self.init_agent()
        
        self.init_variable()
        self.define_subscriber()
        self.define_publisher()
        rospy.loginfo("The agent controller has been initialized!")

    def init_agent(self):
        self.agent = VAECtrl(**model_config)
        self.warm_start()

    def warm_start(self):
        image_size = self.agent.VAE_model.input_dim
        test_color_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        test_state_extra = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        for i in range(5):
            self.agent.predict(test_color_img, state_extra=test_state_extra)

    def init_variable(self):
        # flag
        self.is_active = True
        self.camera_is_ready = False
        self.mavros_is_ready = False

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
        self.camera_timeout_counter = 0
        self.mavros_timeout_counter = 0

    def define_subscriber(self):
        # camera color image
        rospy.Subscriber(
            "/d435i/color/image_raw/",
            Image,
            self.color_image_callback,
            queue_size=5,
            tcp_nodelay=True,
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

    def define_publisher(self):
        self.control_cmd_pub =rospy.Publisher(
            "/my_controller/yaw_cmd",
            ControlCmd,
            queue_size=10,
        )

    def color_image_callback(self, msg):
        if msg is not None:
            self.camera_is_ready = True
            self.last_color_timestamp = msg.header.stamp
            self.color_img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)

    def local_position_odom_callback(self, msg):
        if msg is not None:
            self.mavros_is_ready = True
            self.last_local_position_timestamp = msg.header.stamp
            self.roll, self.pitch, _ = euler_from_quaternion(msg.pose.pose.orientation)

    def velocity_body_callback(self, msg):
        if msg is not None:
            self.body_linear_x = msg.twist.linear.x
            self.body_linear_y = msg.twist.linear.y
            self.body_angular_z = msg.twist.angular.z

    def rc_in_callback(self, msg):
        if msg is not None:
            if msg.channels[6] > 1500: # use channel 7 to control the mode
                if not self.is_active:
                    if (not self.camera_is_ready) or (not self.mavros_is_ready):
                        rospy.logerr("Switched to AI agent control Failed!")
                    else:
                        self.is_active = True
                        rospy.loginfo("Switched to AI agent control!")
            else:
                if self.is_active:
                    self.is_active = False
                    rospy.loginfo("Switched to Manual Control!")

    def publish_control_cmd(self, control_cmd, is_active):
        msg = ControlCmd()
        msg.header.stamp = rospy.Time.now()
        msg.command = constrain_value(control_cmd, -1.0, 1.0)
        msg.is_active = is_active
        self.control_cmd_pub.publish(msg)

    def check_status(self):
        # check topic timeout
        if self.camera_is_ready and (rospy.Time.now() - self.last_color_timestamp).to_sec() > 0.2:
            rospy.logwarn_throttle(1, 'Color image stream rate is slow!')
            self.camera_timeout_counter += 1

        if self.mavros_is_ready and (rospy.Time.now() - self.last_local_position_timestamp).to_sec() > 0.1:
            rospy.logwarn_throttle(1, 'Mavros data stream rate is slow!')
            self.mavros_timeout_counter += 1

        # reset if timeout counter exceed maximum value
        if self.camera_timeout_counter > (15 * 2):
            rospy.logerr("Connection to camera lost!")
            self.reset()

        if self.mavros_timeout_counter > (15 * 2):
            rospy.logerr("Connection to pixhawk lost!")
            self.reset()

    def run(self):
        rospy.loginfo("Start running!")
        while not rospy.is_shutdown():
            # check status
            self.check_status()

            # generate control cmd
            control_cmd = 0.0
            if self.camera_is_ready and self.mavros_is_ready and self.is_active and (self.color_img is not None):
                state_extra = np.array([
                    self.roll,
                    self.pitch,
                    self.body_linear_x,
                    self.body_linear_y,
                    self.body_angular_z,
                ], dtype=np.float32)

                agent_output = self.agent.predict(self.color_img, is_bgr=False, state_extra=state_extra)

                if abs(agent_output) < 1e-2:
                    agent_output = 0.0

                control_cmd = agent_output

            self.publish_control_cmd(control_cmd, self.is_active)
            self.rate.sleep()

    def reset(self):
        self.init_variable()


if __name__ == "__main__":
    handler = AgentControl()
    rospy.sleep(3.0)
    handler.run()
