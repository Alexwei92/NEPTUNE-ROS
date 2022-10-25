import os
import pickle
import math
import rospy
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from std_msgs.msg import Float64
from sensor_msgs.msg import NavSatFix, Image, CompressedImage
from mavros_msgs.msg import HomePosition, GPSRAW
from piksi_rtk_msgs.msg import ReceiverState_V2_4_1

from utils.plot_utils import FieldMapPlot
from utils.navigation_utils import get_local_xy_from_latlon

is_replay = True
use_piksi = True

px4_gps_fix_type = {
    0: 'No GPS',    # No GPS connected
    1: 'No Fix',    # No position information, GPS is connected
    2: '2D Fix',    # 2D position
    3: '3D Fix',    # 3D position
    4: '3D DGPS',   # DGPS/SBAS aided 3D position
    5: 'RTK Float', # TK float, 3D position
    6: 'RTK_Fixed', # TK Fixed, 3D position
    7: 'Static',    # Static fixed, typically used for base stations
    8: 'PPP',
}

class GPSListener():
    LOOP_RATE = 10
    TIME_OUT = 0.5

    def __init__(self, map_handler):
        rospy.init_node('gps_listener', anonymous=True)
        self.rate = rospy.Rate(self.LOOP_RATE)
        self.map_handler = map_handler

        self.init_variables()
        self.define_subscriber()
    
    def init_variables(self):
        # flag
        self.has_initialized = False

        # global position
        self.home_wgs = None
        self.current_lat_piksi = 0.0
        self.current_lon_piksi = 0.0
        self.current_lat_px4 = 0.0
        self.current_lon_px4 = 0.0
        self.compass_heading = 0.0 # rad

        # local position (calculated from lat, lon)
        self.utm_T_local = None
        self.local_x_piksi = 0.0
        self.local_y_piksi = 0.0
        self.local_x_px4 = 0.0
        self.local_y_px4 = 0.0

        # gps raw
        self.num_sat_piksi = 0
        self.fix_type_piksi = 'No GPS'
        self.num_sat_px4 = 0
        self.fix_type_px4 = 'No GPS'

        # image
        self.camera_heartbeat_time = None
        self.color_img = None

    def define_subscriber(self):
        # piksi best fix position
        piksi_best_fix_topic = "/piksi/navsatfix_best_fix"
        rospy.Subscriber(
            piksi_best_fix_topic,
            NavSatFix,
            self.navsatfix_best_fix_callback,
            queue_size=10,
        )

        # piksi receiver state
        rospy.Subscriber(
            '/piksi/debug/receiver_state',
            ReceiverState_V2_4_1,
            self.receiver_state_callback,
            queue_size=10,
        )

        # home position
        home_position_topic = "/mavros/home_position/home"
        rospy.Subscriber(
            home_position_topic,
            HomePosition,
            self.home_position_callback,
            queue_size=10,
        )

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
            queue_size=10,
        )

        # gps status
        gps_status_topic = "/mavros/gpsstatus/gps1/raw"
        rospy.Subscriber(
            gps_status_topic,
            GPSRAW,
            self.gps_status_callback,
            queue_size=10,
        )

        # realsense camera
        if is_replay:
            color_image_topic = "/d435i/color/image_raw/compressed"
            rospy.Subscriber(
                color_image_topic,
                CompressedImage,
                self.color_image_callback,
                queue_size=10,
            )
        else:
            color_image_topic = "/d435i/color/image_raw"
            rospy.Subscriber(
                color_image_topic,
                Image,
                self.color_image_callback,
                queue_size=10,
            )
    
    def set_utm_T_local(self, utm_T_local):
        self.utm_T_local = np.copy(utm_T_local)

    def navsatfix_best_fix_callback(self, msg):
        self.current_lat_piksi = msg.latitude
        self.current_lon_piksi = msg.longitude
        self.current_alt_piksi = msg.altitude

        if self.utm_T_local is not None:
            local_pos_piksi = get_local_xy_from_latlon(
                self.current_lat_piksi,
                self.current_lon_piksi,
                self.utm_T_local
            )
            self.local_x_piksi = local_pos_piksi[0]
            self.local_y_piksi = local_pos_piksi[1]

    def receiver_state_callback(self, msg):
        self.num_sat_piksi = msg.num_sat
        self.fix_type_piksi = msg.fix_mode

    def home_position_callback(self, msg):
        if self.home_wgs is None:
            self.home_wgs = (
                msg.geo.latitude,
                msg.geo.longitude,
                msg.geo.altitude,
            )
            rospy.loginfo("home position: " + str(self.home_wgs))

    def global_position_callback(self, msg):
        self.current_lat_px4 = msg.latitude
        self.current_lon_px4 = msg.longitude
        self.current_alt = msg.altitude

        if self.utm_T_local is not None:
            local_pos_px4 = get_local_xy_from_latlon(
                self.current_lat_px4,
                self.current_lon_px4,
                self.utm_T_local
            )
            self.local_x_px4 = local_pos_px4[0]
            self.local_y_px4 = local_pos_px4[1]

    def compass_callback(self, msg):
        """
        Coordinate frame: Down (z), North is zero
        """
        heading = msg.data
        self.compass_heading = math.radians(heading)

    def gps_status_callback(self, msg):
        self.num_sat_px4 = msg.satellites_visible
        self.fix_type_px4 = px4_gps_fix_type[msg.fix_type]

    def color_image_callback(self, msg):
        if is_replay:
            color_img = np.frombuffer(msg.data, dtype=np.uint8)
            color_img = cv2.imdecode(color_img, cv2.IMREAD_COLOR) # RGB
            self.color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        else:
            self.color_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        # self.color_img = cv2.resize(color_img, (320,240))
        self.camera_heartbeat_time = msg.header.stamp

    def run(self):
        while not rospy.is_shutdown() and self.has_initialized:
            # tic = time.time()
            current_heading = -self.compass_heading + math.pi/2
            print(current_heading)
            # Update graph
            if self.map_handler:
                if use_piksi:
                    self.map_handler.update_graph(
                        pos=[self.local_x_piksi, self.local_y_piksi],
                        heading=current_heading,
                        num_sat=self.num_sat_piksi,
                        fix_type=self.fix_type_piksi,
                        pos2=[self.local_x_px4, self.local_y_px4],
                        num_sat2=self.num_sat_px4,
                        fix_type2=self.fix_type_px4,
                    )
                else:
                    self.map_handler.update_graph(
                        pos=[self.local_x_px4, self.local_y_px4],
                        heading=current_heading,
                        num_sat=self.num_sat_px4,
                        fix_type=self.fix_type_px4,
                        pos2=[self.local_x_piksi, self.local_y_piksi],
                        num_sat2=self.num_sat_piksi,
                        fix_type2=self.fix_type_piksi,
                    )

                if not is_replay:
                    if self.camera_heartbeat_time is not None:
                        if (rospy.Time.now() - self.camera_heartbeat_time).to_sec() > self.TIME_OUT:
                            self.color_img = None
                            rospy.logerr_throttle(2, "Camera lost!")
                self.map_handler.update_image(self.color_img)
                
                plt.pause(1e-5)

            self.rate.sleep()
            # print(1/(time.time()-tic))


if __name__ == "__main__":
    
    curr_dir = os.path.dirname(__file__)
    data_path = os.path.join(curr_dir, "ground_truth/plant_field.pkl")

    # Configure map
    with open(data_path, 'rb') as file:
        data = pickle.load(file)

    field_bound = {
        'latlon': data['field_bound_latlon'],
        'local': data['field_bound_local'],
    }

    map_handler = FieldMapPlot(
        data['row_data'],
        field_bound,
    )

    print('Load field data successfully!')

    # GPS Navigation
    handler = GPSListener(map_handler)
    handler.set_utm_T_local(data['utm_T_local'])
    handler.has_initialized = True

    # Sleep for 1 second
    rospy.sleep(1.0)

    handler.run()    