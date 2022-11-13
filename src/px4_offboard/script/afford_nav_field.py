#!/usr/bin/env python3
import os
import pickle
import math
import rospy
import numpy as np
from shapely.geometry.polygon import Polygon

from std_msgs.msg import Float64
from sensor_msgs.msg import NavSatFix, Image
from mavros_msgs.msg import GPSRAW
from px4_offboard.msg import Affordance, ControlCmd
from mavros_msgs.srv import SetMode

from utils.navigation_utils import (
    get_local_xy_from_latlon,
    get_projection_point2line,
    whether_in_polygon,
)
from utils.math_utils import wrap_2PI, wrap_PI, constrain_value

# px4_gps_fix_type = {
#     0: 'No GPS',    # No GPS connected
#     1: 'No Fix',    # No position information, GPS is connected
#     2: '2D Fix',    # 2D position
#     3: '3D Fix',    # 3D position
#     4: '3D DGPS',   # DGPS/SBAS aided 3D position
#     5: 'RTK Float', # TK float, 3D position
#     6: 'RTK_Fixed', # TK Fixed, 3D position
#     7: 'Static',    # Static fixed, typically used for base stations
#     8: 'PPP',
# }

MAX_YAWRATE = 45 # rad/s

def calc_affordance_cmd(affordance, max_yawrate=45, flag=0):
    if affordance is None:
        return 0.0

    dist_center = affordance['dist_center']
    rel_angle = -affordance['rel_angle']

    # dist_noise = np.random.random() * (1.5 * 2) - 1.5
    # dist_center += dist_noise
    # angle_noise = np.random.random() * (0.3 * 2) - 0.3
    # rel_angle += angle_noise

    if flag == 0: # Option 1: Sigmoid function
        angle_gain = 15 # (higher is more responsive)
        dist_gain = 10
        cmd = 1.0 * (2 / (1 + math.exp(angle_gain * rel_angle/(math.pi/2) + dist_gain * dist_center/6.0)) - 1)

    elif flag == 1: # Option 2: Stanley
        control_gain = 5.0
        head_gain = 5.0

        theta_e = head_gain * (-rel_angle)
        theta_d = np.arctan2(control_gain * (-dist_center), 0.8)
        stanley_output = theta_e + theta_d
        cmd = 1.0 * stanley_output * (15) / max_yawrate

    else:
        cmd = 0.0

    return constrain_value(cmd, -1.0, 1.0)


class AffordanceNav():
    LOOP_RATE = 15
    TIME_OUT = 0.5

    STATUS_IDLE = 0
    STATUS_TRACKING = 1
    STATUS_EMERGENCY = 2

    def __init__(self):
        rospy.init_node('affordance_navigation')
        self.rate = rospy.Rate(self.LOOP_RATE)
        target_index = rospy.get_param("~target_index", 4)
        rospy.wait_for_service("/mavros/set_mode")

        self.init_variables()
        self.define_subscriber()
        self.define_publisher()
        self.load_field_map()
        self.set_target_index(target_index)
    
    def init_variables(self):
        # flag
        self.has_initialized = False
        self.target_index = None
        self.status = self.STATUS_IDLE

        # global position
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.compass_heading = 0.0 # rad

        # local position (calculated from lat, lon)
        self.utm_T_local = None
        self.local_x = 0.0
        self.local_y = 0.0

        # gps raw
        self.num_sat = 0
        self.fix_type = 0

        # image
        self.camera_heartbeat_time = None
        self.color_img = None

        # affordance
        self.affordance = {}
        self.last_cmd = 0.0

    def define_subscriber(self):
        # global position
        global_position_topic = "/mavros/global_position/global"
        rospy.Subscriber(
            global_position_topic,
            NavSatFix,
            self.global_position_callback,
            queue_size=10,
        )

        # compass
        compass_topic = "/mavros/global_position/compass_hdg"
        rospy.Subscriber(
            compass_topic,
            Float64,
            self.compass_callback,
            queue_size=5,
        )

        # gps status
        gps_status_topic = "/mavros/gpsstatus/gps1/raw"
        rospy.Subscriber(
            gps_status_topic,
            GPSRAW,
            self.gps_status_callback,
            queue_size=5,
        )

        # realsense camera
        color_image_topic = "/d435i/color/image_raw"
        rospy.Subscriber(
            color_image_topic,
            Image,
            self.color_image_callback,
            queue_size=5,
        )
    
    def define_publisher(self):
        # yaw control command
        self.cmd_pub = rospy.Publisher(
            "/my_controller/yaw_cmd",
            ControlCmd,
            queue_size=5,
        )

        # estimated affordance
        self.afford_pub = rospy.Publisher(
            "/estimated_affordance",
            Affordance,
            queue_size=5,
        )

    def load_field_map(self):
        curr_dir = os.path.dirname(__file__)
        data_path = os.path.join(curr_dir, "ground_truth/plant_field.pkl")

        # Configure map
        with open(data_path, 'rb') as file:
            field_data = pickle.load(file)

        self.field_data = field_data['row_data']
        self.set_utm_T_local(field_data['utm_T_local'])

    def set_utm_T_local(self, utm_T_local):
        self.utm_T_local = np.copy(utm_T_local)

    def set_target_index(self, target_index):
        self.target_index = target_index
        self.target_polygon = Polygon(
            [vertex for vertex in self.field_data[target_index]['vertice']]
        )
        rospy.loginfo("Row Index: %i" % target_index)

    def global_position_callback(self, msg):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        self.current_alt = msg.altitude

        if self.utm_T_local is not None:
            local_pos = get_local_xy_from_latlon(
                self.current_lat,
                self.current_lon,
                self.utm_T_local
            )
            self.local_x = local_pos[0]
            self.local_y = local_pos[1]

            if not self.has_initialized and self.target_index is not None:
                field_data_current = self.field_data[self.target_index]
                if self.local_x > field_data_current['treelines_actual'][0][1][0]:
                    self.direction = -1
                else:
                    self.direction = 1
                rospy.loginfo("Flying Direction: %i" % self.direction)

                reference_line = field_data_current['centerline_actual']
                reference_heading = np.arctan2(
                    field_data_current['centerline_actual'][1][1] - field_data_current['centerline_actual'][0][1],
                    field_data_current['centerline_actual'][1][0] - field_data_current['centerline_actual'][0][0],
                )
                if self.direction < 0:
                    reference_line = np.flip(reference_line, axis=0)
                    reference_heading += np.pi
                self.reference_line = reference_line
                self.reference_heading = wrap_2PI(reference_heading)
                self.total_length = np.sqrt(
                    (field_data_current['centerline_actual'][0][0] - field_data_current['centerline_actual'][1][0])**2
                    + (field_data_current['centerline_actual'][0][1] - field_data_current['centerline_actual'][1][1])**2 
                )
                self.has_initialized = True
                rospy.loginfo("Ready to Go!")

    def compass_callback(self, msg):
        """
        Coordinate frame: Down (z), North is zero
        """
        heading = msg.data
        self.compass_heading = math.radians(heading)

    def gps_status_callback(self, msg):
        self.num_sat = msg.satellites_visible
        self.fix_type = msg.fix_type
        if self.fix_type < 3:
            rospy.logwarn_throttle(1, "GPS is not Fixed!")

    def color_image_callback(self, msg):
        self.color_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.camera_heartbeat_time = msg.header.stamp

    def publish_affordance(self):
        afford = Affordance()
        afford.header.stamp = rospy.Time.now()
        afford.dist_center = self.affordance['dist_center']
        afford.rel_angle = self.affordance['rel_angle']
        afford.in_bound = self.affordance['in_bound']
        self.afford_pub.publish(afford)

    def publish_control_cmd(self, cmd):
        control_cmd = ControlCmd()
        control_cmd.header.stamp = rospy.Time.now()
        control_cmd.command = cmd
        self.cmd_pub.publish(control_cmd)

    def raise_emergency(self):
        self.status = self.STATUS_EMERGENCY
        rospy.logwarn("Emergency Stop! Swithched to POSCTL Mode!")

    def run(self):
        while not rospy.is_shutdown() and self.has_initialized:
            if self.status == self.STATUS_EMERGENCY:
                break
        
            current_heading = wrap_2PI(-self.compass_heading + math.pi/2)
            
            lat_proj, lon_proj = get_projection_point2line(
                [self.local_x, self.local_y],
                self.reference_line,
            )
            rel_angle = wrap_PI(current_heading - self.reference_heading)
            in_bound = whether_in_polygon(
                [self.local_x, self.local_y, current_heading],
                self.target_polygon,
            )
            self.affordance['dist_center'] = lat_proj
            self.affordance['rel_angle'] = rel_angle
            self.affordance['in_bound'] = in_bound
            
            if lon_proj > self.total_length + 1.5:
                set_mode_proxy = rospy.ServiceProxy("/mavros/set_mode", SetMode)
                set_mode_proxy(custom_mode = "POSCTL")
                rospy.loginfo("Reached end point!")
                break

            if not in_bound:
                if self.status == self.STATUS_TRACKING:
                    self.raise_emergency()
                else:
                    self.status = self.STATUS_IDLE
                    self.last_cmd = 0.0
                    self.affordance['in_bound'] = True
            else:
                if self.status == self.STATUS_IDLE:
                    rospy.loginfo("Start Tracking") 
                    self.status = self.STATUS_TRACKING
                
                cmd = calc_affordance_cmd(self.affordance)

                # LPF
                alpha = 1.0
                cmd = alpha * cmd + (1 - alpha) * self.last_cmd

                self.last_cmd = cmd
                self.publish_control_cmd(cmd)
        
            self.publish_affordance()
            self.rate.sleep()

if __name__ == "__main__":
    # Affordance-based Controller
    controller = AffordanceNav()

    # Sleep for 1 second
    rospy.sleep(1.0)

    controller.run()