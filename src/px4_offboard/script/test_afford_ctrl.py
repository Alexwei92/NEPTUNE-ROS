#!/usr/bin/env python
import os
import time
import rospy
from std_msgs.msg import Float32, Float64
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from mavros_msgs.msg import HomePosition
from mavros_msgs.srv import SetMode

import matplotlib.pyplot as plt
plt.style.use('dark_background')

from math_utils import *
from plot_utils import *
from affordance_utils import *
from px4_offboard.msg import Affordance

import warnings
warnings.filterwarnings('ignore')

MAX_YAWRATE = 45 # rad/s

def calc_affordance_cmd(affordance):
    if not affordance:
        return 0.0

    if 'dist_center_width' in affordance:
        dist_center_width = affordance['dist_center_width'] 
        dist_left_width = affordance['dist_left_width'] - 0.5
    else:
        dist_center_width = affordance['dist_center'] / (affordance['dist_left'] + affordance['dist_right'])
        dist_left_width = affordance['dist_left'] / (affordance['dist_left'] + affordance['dist_right']) - 0.5
    
    rel_angle = affordance['rel_angle']

    # if abs(rel_angle) < 3 / 180 * math.pi:
    #     rel_angle = 0

    # if abs(rel_angle) < 0.03:
    #     dist_center_width = 0

    # Option 1: Sigmoid function
    # cmd = 1.0 * (2 /(1 + math.exp(15*(1.5*rel_angle/(math.pi/2) + 1.0*dist_center_width))) - 1)
    
    # Option 2: Stanley
    stanley_cmd = -rel_angle + math.atan(-2.5 * affordance['dist_center'] / 1.5)
    cmd = stanley_cmd * (15) / MAX_YAWRATE

    return constrain_float(cmd, -1.0, 1.0), affordance['in_bound']

class AffordanceNav():
    LOOP_RATE = 15

    def __init__(self, map_handler, xy_offset):
        rospy.init_node('map_listener', anonymous=True)
        self.rate = rospy.Rate(self.LOOP_RATE)

        self.map_handler = map_handler
        self.xy_offset = xy_offset
        self.init_variables()

        rospy.wait_for_service("/mavros/set_mode")
        self.define_subscriber()
        self.define_publisher()

        rospy.loginfo(
            rospy.get_name() + ": Start!"
        )
    
    def init_variables(self):
        # global position
        self.home_utm = None
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0 # TODO
        self.compass_heading = 0.0

        # local position
        self.current_x = 0
        self.current_y = 0
        self.current_heading = 0
        self.last_cmd = 0
        self.affordance = None

    def define_subscriber(self):
        # home position
        home_position_topic = "/mavros/home_position/home"
        rospy.Subscriber(
            home_position_topic,
            HomePosition,
            self.home_position_callback,
            queue_size=1,
        )

        # global position
        global_position_topic = "/mavros/global_position/global"
        rospy.Subscriber(
            global_position_topic,
            NavSatFix,
            self.global_position_callback,
            queue_size=5,
        )

        # compass
        compass_topic = "/mavros/global_position/compass_hdg"
        rospy.Subscriber(
            compass_topic,
            Float64,
            self.compass_callback,
            queue_size=5,
        )

        # local odom in global frame
        local_odom_in_global_topic = "/mavros/global_position/local"
        rospy.Subscriber(
            local_odom_in_global_topic,
            Odometry,
            self.local_odom_in_global_callback,
            queue_size=5,
        )

        # local pose in local frame
        local_pose_in_local_topic = "/mavros/local_position/pose"
        rospy.Subscriber(
            local_pose_in_local_topic,
            PoseStamped,
            self.localPose_callback,
            queue_size=5,
        )

    def define_publisher(self):
        # yaw control command
        self.cmd_pub = rospy.Publisher(
            "/my_controller/yaw_cmd", 
            Float32,
            queue_size=5,
        )

        # estimated affordance
        self.afford_pub = rospy.Publisher(
            "/estimated_affordance",
            Affordance,
            queue_size=5,
        )

    def load_spline_data(self, map_path):
        self.map_data = read_map_data(map_path)

    def run(self):
        while not rospy.is_shutdown():
            # Update graph
            if self.map_handler:
                self.map_handler.update_graph([self.current_x, self.current_y], self.current_heading)
                plt.pause(1e-5)

            # Affordance
            if self.affordance:
                self.publish_affordance()

            # Run rule-based controller
            cmd, in_bound = calc_affordance_cmd(self.affordance)
            if not in_bound:
                cmd = 0.0
                set_mode_proxy = rospy.ServiceProxy("/mavros/set_mode", SetMode)
                set_mode_proxy(custom_mode = "POSCTL")
                rospy.logwarn_throttle(
                    2, "Fly out of bound! Be caution!"
                )
                    
            else:
                # apply a low-pass filter
                alpha = 1.0
                cmd = alpha * cmd + (1 - alpha) * self.last_cmd
            
            self.last_cmd = cmd
            self.cmd_pub.publish(Float32(cmd))
            self.rate.sleep()

    def home_position_callback(self, msg):
        if self.home_utm is None:
            self.home_utm = msg.geo

    def global_position_callback(self, msg):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        self.current_alt = msg.altitude
        status = msg.status.status
        # print(self.current_lat, self.current_lon)

    def compass_callback(self, msg):
        heading = msg.data
        self.compass_heading = (heading / 180 * math.pi)

    def local_odom_in_global_callback(self, msg):
        pose = msg.pose
        twist = msg.twist
        pass

    def localPose_callback(self, msg):
        current_pose = msg.pose
        self.current_x = current_pose.position.x + self.xy_offset[0]
        self.current_y = current_pose.position.y + self.xy_offset[1]
        _, _, yaw = euler_from_quaternion(current_pose.orientation)
        self.current_heading = wrap_2PI(yaw)

        # Calculate affordance
        pose = {
            'pos': [self.current_x, self.current_y],
            'yaw': self.current_heading,
            'direction': self.map_handler.get_direction() if self.map_handler else 1,
        }
        self.affordance = calculate_affordance(self.map_data, pose)
        print(self.compass_heading, self.current_heading)

    def publish_affordance(self):
        afford = Affordance()
        afford.header.stamp = rospy.Time.now()
        afford.dist_center = self.affordance['dist_center']
        afford.dist_left = self.affordance['dist_left']
        afford.dist_right = self.affordance['dist_right']
        afford.rel_angle = self.affordance['rel_angle']
        afford.in_bound = self.affordance['in_bound']
        self.afford_pub.publish(afford)

    def reset(self):
        self.init_variables()

    def get_home_utm(self):
        return self.home_utm


if __name__ == '__main__':

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(curr_dir, 'spline_result/spline_result.csv')
   
    # Configure map
    map_origin = (38.588156, -121.705512) # origin GPS UTM
    map_handler = MapPlot(map_path)
    print("Configured the map successfully!")

    # locate the drone on the map
    x_ratio = 1 / 9 * 24.22
    y_ratio = 1 / 5 * 30.86
    takeoff_location = [27.171 * x_ratio, 2.72 * y_ratio]

    # ROS node
    handler = AffordanceNav(map_handler, takeoff_location)
    handler.load_spline_data(map_path)

    # test
    home_utm = handler.get_home_utm()
    
    

    # Sleep for 1 second
    tic = time.perf_counter()
    while time.perf_counter() - tic < 1.0:
        pass

    handler.run()