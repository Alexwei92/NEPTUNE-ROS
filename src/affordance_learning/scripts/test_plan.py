import os
import pickle
import math
import rospy
import matplotlib.pyplot as plt
import numpy as np

from std_msgs.msg import Float32, Float64
from sensor_msgs.msg import NavSatFix
from mavros_msgs.msg import HomePosition

from utils.plot_utils import FieldMapPlot
from utils.navigation_utils import get_local_xy_from_latlon

class GPSListener():
    LOOP_RATE = 15

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
        self.local_x = 0.0
        self.local_y = 0.0

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

    def run(self):
        while not rospy.is_shutdown() and self.has_initialized:
            current_heading = -self.compass_heading + math.pi/2
            
            # Update graph
            if self.map_handler:
                if self.frame == 'latlon':
                    self.map_handler.update_graph(
                        [self.current_lon, self.current_lat],
                        current_heading,
                    )
                if self.frame == 'local':
                    self.map_handler.update_graph(
                        [self.local_x, self.local_y],
                        current_heading,
                    )
                plt.pause(1e-5)

            self.rate.sleep()


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