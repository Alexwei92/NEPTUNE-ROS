import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, Pose
import matplotlib.pyplot as plt

from utils import *

loop_rate = 15 # Hz

def affordance_ctrl(affordance):
    if 'dist_center_width' in affordance:
        dist_center_width = affordance['dist_center_width'] 
        dist_left_width = affordance['dist_left_width'] - 0.5
    else:
        dist_center_width = affordance['dist_center'] / (affordance['dist_left'] + affordance['dist_right'])
        dist_left_width = affordance['dist_left'] / (affordance['dist_left'] + affordance['dist_right']) - 0.5
    rel_angle = affordance['rel_angle'] / (np.pi/2)
    
    if abs(rel_angle) < 5 / 180 * math.pi:
        rel_angle = 0

    if abs(rel_angle) < 0.05:
        dist_center_width = 0

    # Sigmoid function
    # cmd = 1.0 * (2 /(1 + math.exp(20*(1.5*rel_angle/np.pi + 3.0*dist_center_width))) - 1)
    cmd = -rel_angle + math.atan(50*dist_center_width)
    
    return cmd

class ros_handler():

    def __init__(self, 
                map_handler,
                offset):
        rospy.init_node('map_listener', anonymous=True)
        self.rate = rospy.Rate(loop_rate)

        self.map_handler = map_handler
        self.offset = offset

        self.current_pose = Pose()
        self.pos_x = 0
        self.pos_y = 0
        self.heading = 0
        self.last_cmd = 0

        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, 
                        self.localPose_callback, queue_size=5)

        self.cmd_pub = rospy.Publisher("/my_controller/yaw_cmd", 
                        Float32, queue_size=5)

    
    def read_spline_data(self, map_path):
        self.map_data = read_map_data(map_path)


    def run(self):
        while not rospy.is_shutdown():
            self.map_handler.update([self.pos_x, self.pos_y], self.heading)
            plt.pause(1e-5)

            cmd = affordance_ctrl(self.affordance)

            # Apply a filter
            alpha = 0.3
            cmd = alpha * cmd + (1 - alpha) * self.last_cmd
            self.cmd_pub.publish(Float32(cmd))
            self.last_cmd = cmd
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
            'direction': 1
        }
        self.affordance = calculate_affordance(self.map_data, pose)
        # print(affordance)


if __name__ == '__main__':
    map_path = 'spline_result/spline_result.csv'
    x_ratio = 1 / 9 * 24.22
    y_ratio = 1 / 5 * 30.86
    takeoff_location = [27.171 * x_ratio, 2.72 * y_ratio]
    
    # Configure map
    map_handler = MapPlot(map_path)
    map_handler.update_start_point(takeoff_location)
    map_handler.update(takeoff_location, 0)
    print("Configured the map successfully!")

    # ROS node
    handler = ros_handler(map_handler, takeoff_location)
    handler.read_spline_data(map_path)

    tic = rospy.Time.now()
    while rospy.Time.now() - tic < rospy.Duration(1.0):
        pass

    handler.run()
