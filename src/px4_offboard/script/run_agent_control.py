#!/usr/bin/env python3
import os
import rospy
import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, PoseStamped
from sensor_msgs.msg import Image
from mavros_msgs.msg import RCIn, HomePosition
from px4_offboard.msg import ControlCmd

from controller import VAELatentController, VAELatentController_TRT
from utils.math_utils import euler_from_quaternion, constrain_value

######################
curr_dir = os.path.dirname(os.path.abspath(__file__))
model_weight_dir = os.path.abspath(os.path.join(curr_dir, "../../../model_weight/vae"))
extra_dir = os.path.abspath(os.path.join(curr_dir, "../../network_training/scripts"))

model_config = {
    'model_weight_path': os.path.join(model_weight_dir, 'combined_vae_latent_ctrl_z_1000.pt'),
    'tensorrt_engine_path': os.path.join(model_weight_dir, 'combined_vae_latent_ctrl_z_1000.trt'),
}
######################

class AgentControl():
    def __init__(self, use_tensorrt=False):
        rospy.init_node("agent_control")
        self.rate = rospy.Rate(15)

        self.init_agent(use_tensorrt)
        self.init_variable()
        self.define_subscriber()
        self.define_publisher()

        rospy.wait_for_message("/mavros/home_position/home", HomePosition)
        rospy.loginfo("The agent controller has been initialized!")

    def init_agent(self, use_tensorrt):
        if use_tensorrt:
            self.agent = VAELatentController_TRT(**model_config)
        else:
            self.agent = VAELatentController(**model_config)
        
        self.use_tensorrt = use_tensorrt
        self.warm_start()

    def warm_start(self, iter=3):
        image_size = self.agent.input_dim
        test_color_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        test_state_extra = np.zeros((6,), dtype=np.float32)
        for _ in range(int(iter)):
            self.agent.predict(test_color_img, state_extra=test_state_extra)

    def init_variable(self):
        # flag
        self.is_active = False
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

        # home position
        self.home_pos_z = 0.0

        # relative height
        self.relative_height = 0.0

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

        # home position
        rospy.Subscriber(
            "/mavros/home_position/home",
            HomePosition,
            self.home_position_callback,
            queue_size=1,
        )

        # local odom
        rospy.Subscriber(
            "/mavros/local_position/pose",
            PoseStamped,
            self.local_position_pose_callback,
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

    def home_position_callback(self, msg):
        if msg is not None:
            if msg.position.z != self.home_pos_z:
                self.home_pos_z = msg.position.z
                rospy.loginfo("Home position updated!")

    def local_position_pose_callback(self, msg):
        if msg is not None:
            self.mavros_is_ready = True
            self.last_local_position_timestamp = msg.header.stamp
            self.roll, self.pitch, _ = euler_from_quaternion(msg.pose.orientation)
            self.relative_height =  msg.pose.position.z - self.home_pos_z

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
        if self.camera_is_ready and (rospy.Time.now() - self.last_color_timestamp).to_sec() > 0.3:
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
                    self.relative_height,
                ], dtype=np.float32)

                agent_output = self.agent.predict(self.color_img, is_bgr=False, state_extra=state_extra)
                # print(agent_output)

                # remove trunction error
                if abs(agent_output) < 1e-2:
                    agent_output = 0.0

                control_cmd = agent_output

            self.publish_control_cmd(control_cmd, self.is_active)
            self.rate.sleep()

    def reset(self):
        self.init_variable()
        rospy.loginfo('Reset!')

if __name__ == "__main__":
    handler = AgentControl(use_tensorrt=True)
    rospy.sleep(2.0)
    rospy.loginfo("Start running!")
    handler.run()
