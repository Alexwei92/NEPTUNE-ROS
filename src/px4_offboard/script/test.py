import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, Pose
from mavros_msgs.srv import SetMode

import matplotlib.pyplot as plt
plt.style.use('dark_background')

from math_utils import *
from plot_utils import *
from affordance_utils import *
from px4_offboard.msg import Affordance

import warnings
warnings.filterwarnings('ignore')

loop_rate = 15 # Hz
MAX_YAWRATE = 45 # rad/s

def affordance_ctrl(affordance):
    if not affordance:
        return 0

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

    # Sigmoid function
    cmd = 1.0 * (2 /(1 + math.exp(15*(1.5*rel_angle/(math.pi/2) + 1.0*dist_center_width))) - 1)
    
    # stanley_cmd = -rel_angle + math.atan(-2.5 * affordance['dist_center'] / 1.5)
    # cmd = stanley_cmd * (15) / MAX_YAWRATE
    return constrain_float(cmd, -1.0, 1.0), affordance['in_bound']

class ros_handler():

    def __init__(self, 
                map_handler,
                offset):
        rospy.init_node('map_listener', anonymous=True)
        self.rate = rospy.Rate(loop_rate)

        self.map_handler = map_handler
        self.offset = offset

        self.reset()

        rospy.wait_for_service("/mavros/set_mode")

        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, 
                        self.localPose_callback, queue_size=5)

        self.cmd_pub = rospy.Publisher("/my_controller/yaw_cmd", 
                        Float32, queue_size=5)

        self.afford_pub = rospy.Publisher("/estimated_affordance",
                        Affordance, queue_size=5)
    
    def read_spline_data(self, map_path):
        self.map_data = read_map_data(map_path)

    def run(self):
        while not rospy.is_shutdown():
            # Update graph
            if self.map_handler:
                self.map_handler.update_graph([self.pos_x, self.pos_y], self.heading)
                plt.pause(1e-5)

            # Affordance
            if self.affordance:
                afford = Affordance()
                afford.header.stamp = rospy.Time.now()
                afford.dist_center = self.affordance['dist_center']
                afford.dist_left = self.affordance['dist_left']
                afford.dist_right = self.affordance['dist_right']
                afford.rel_angle = self.affordance['rel_angle']
                afford.in_bound = self.affordance['in_bound']
                self.afford_pub.publish(afford)

            # Run rule-based controller
            cmd, in_bound = affordance_ctrl(self.affordance)
            if not in_bound:
                cmd = 0
                # set_mode_proxy = rospy.ServiceProxy("/mavros/set_mode", SetMode)
                # set_mode_proxy(custom_mode = "POSCTL")
                print("Fly out of bound! Be caution!")
                    
            else:
                alpha = 1.0
                cmd = alpha * cmd + (1 - alpha) * self.last_cmd
            self.last_cmd = cmd
            self.cmd_pub.publish(Float32(cmd))
            
            self.rate.sleep()

    def localPose_callback(self, msg):
        self.current_pose = msg.pose
        self.pos_x = self.current_pose.position.x + self.offset[0]
        self.pos_y = self.current_pose.position.y + self.offset[1]
        _, _, yaw = euler_from_quaternion(self.current_pose.orientation)
        self.heading = wrap_2PI(yaw)

        # Calculate affordance
        pose = {
            'pos': [self.pos_x, self.pos_y],
            'yaw': self.heading,
            'direction': self.map_handler.get_direction() if self.map_handler else 1,
        }
        self.affordance = calculate_affordance(self.map_data, pose)

    def reset(self):
        self.current_pose = Pose()
        self.pos_x = 0
        self.pos_y = 0
        self.heading = 0
        self.last_cmd = 0
        self.affordance = None


if __name__ == '__main__':
    map_path = 'spline_result/spline_result.csv'
    x_ratio = 1 / 9 * 24.22
    y_ratio = 1 / 5 * 30.86
    takeoff_location = [27.171 * x_ratio, 2.72 * y_ratio]
    
    # Configure map
    map_handler = MapPlot(map_path)
    print("Configured the map successfully!")

    # ROS node
    handler = ros_handler(map_handler, takeoff_location)
    handler.read_spline_data(map_path)

    # Sleep for 1 second
    tic = rospy.Time.now()
    while rospy.Time.now() - tic < rospy.Duration(1.0):
        pass

    handler.run()