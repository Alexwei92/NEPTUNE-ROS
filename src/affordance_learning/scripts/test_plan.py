import os
import pickle
import math
import rospy
import matplotlib.pyplot as plt
import numpy as np
import time
from std_msgs.msg import Float32, Float64
from sensor_msgs.msg import NavSatFix, Image, CompressedImage
from mavros_msgs.msg import HomePosition, GPSRAW

from utils.plot_utils import FieldMapPlot
from utils.navigation_utils import get_local_xy_from_latlon


class GPSListener():
    LOOP_RATE = 10
    TIME_OUT = 0.3

    def __init__(self, map_handler):
        rospy.init_node('gps_listener', anonymous=True)
        self.rate = rospy.Rate(self.LOOP_RATE)
        self.map_handler = map_handler
        self.frame = self.map_handler.frame

        self.init_variables()
        self.define_subscriber()
    
    def init_variables(self):
        # flag
        self.has_initialized = False

        # global position
        self.home_wgs = None
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
            queue_size=1,
        )

        # compass
        compass_topic = "/mavros/global_position/compass_hdg"
        rospy.Subscriber(
            compass_topic,
            Float64,
            self.compass_callback,
            queue_size=1,
        )

        # gps status
        gps_status_topic = "/mavros/gpsstatus/gps1/raw"
        rospy.Subscriber(
            gps_status_topic,
            GPSRAW,
            self.gps_status_callback,
            queue_size=1,
        )

        # realsense camera
        color_image_topic = "/d435i/color/image_raw"
        rospy.Subscriber(
            color_image_topic,
            Image,
            self.color_image_callback,
            queue_size=1,
        )
    
    def set_utm_T_local(self, utm_T_local):
        self.utm_T_local = np.copy(utm_T_local)

    def home_position_callback(self, msg):
        if self.home_wgs is None:
            self.home_wgs = (
                msg.geo.latitude,
                msg.geo.longitude,
                msg.geo.altitude,
            )
            rospy.loginfo("home position: " + str(self.home_wgs))

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

    def compass_callback(self, msg):
        """
        Coordinate frame: Down (z), North is zero
        """
        heading = msg.data
        self.compass_heading = math.radians(heading)

    def gps_status_callback(self, msg):
        self.num_sat = msg.satellites_visible
        self.fix_type = msg.fix_type

    def color_image_callback(self, msg):
        self.color_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        # self.color_img = cv2.resize(color_img, (320,240))
        self.camera_heartbeat_time = msg.header.stamp

    def run(self):
        while not rospy.is_shutdown() and self.has_initialized:
            # tic = time.time()
            current_heading = -self.compass_heading + math.pi/2
            # Update graph
            if self.map_handler:
                if self.frame == 'latlon':
                    self.map_handler.update_graph(
                        pos=[self.current_lon, self.current_lat],
                        heading=current_heading,
                        num_sat=self.num_sat,
                        fix_type=self.fix_type,
                    )
                if self.frame == 'local':
                    self.map_handler.update_graph(
                        pos=[self.local_x, self.local_y],
                        heading=current_heading,
                        num_sat=self.num_sat,
                        fix_type=self.fix_type,
                    )

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
        data['column_data'],
        field_bound,
        frame='local',
    )

    print('Load field data successfully!')

    # GPS Navigation
    handler = GPSListener(map_handler)
    handler.set_utm_T_local(data['utm_T_local'])
    handler.has_initialized = True

    # Sleep for 1 second
    rospy.sleep(1.0)

    handler.run()    