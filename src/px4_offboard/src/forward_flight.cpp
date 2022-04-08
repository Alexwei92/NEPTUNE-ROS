#include <ros/ros.h>
#include <math.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Header.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/RCIn.h>
#include <mavros_msgs/ManualControl.h>
#include <mavros_msgs/State.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <boost/bind.hpp>

#include "common.h"
#include "math_utils.h"

#define LOOP_RATE_DEFAULT   30 // Hz

class ForwardCtrl
{
public:
    ForwardCtrl()
    {
        InitializeTarget();
        
        // Publisher
        target_setpoint_pub = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);  

        // Subscriber
        state_sub = nh.subscribe<mavros_msgs::State>("/mavros/state", 5, 
                &ForwardCtrl::StateCallback, this);
        local_pose_sub = nh.subscribe<geometry_msgs::PoseStamped>("/mavros/local_position/pose", 5,
                &ForwardCtrl::LocalPoseCallback, this);

        #ifdef SITL_MODE
        rcin_sub = nh.subscribe<mavros_msgs::ManualControl>("/mavros/manual_control/control", 5, 
                boost::bind(&ForwardCtrl::JoystickCallback, this, _1));
        #else
        rcin_sub = nh.subscribe<mavros_msgs::RCIn>("/mavros/rc/in", 5, 
                boost::bind(&ForwardCtrl::RCInCallback, this, _1, YAW_CHANNEL));
        #endif
    }

    void run()
    {
        ros::Rate loop_rate(LOOP_RATE_DEFAULT);
        ROS_INFO("Node Started!");
        while (ros::ok()) {
            target.header.stamp = ros::Time::now();
            target.header.seq++;

            if (current_state.mode == "OFFBOARD") { 
                // 2 second fade in
                double ratio = 1.0;
                double dt = (ros::Time::now() - offboard_start_time).toSec();
                if (dt < 2.0) { 
                    ratio = dt / 2.0;
                }                
                // velocity
                geometry_msgs::Vector3 velocity_local;
                velocity_local.x = FORWARD_SPEED * ratio;
                velocity_local.y = 0; 
                velocity_local.z = 0;
                if (target.coordinate_frame == 1) {
                    rotate_body_frame_to_NE(velocity_local.x, velocity_local.y, yaw_rad);
                }
                target.velocity = velocity_local;
                // yaw rate
                target.yaw_rate = -yaw_cmd * MAX_YAWRATE * DEG2RAD;

            }
            
            target_setpoint_pub.publish(target);
            ros::spinOnce();
            loop_rate.sleep();
        }
        ROS_INFO("Stop Offboard Mode!");
    }

private:
    void InitializeTarget() {
        // bitmask
        target.header.frame_id = "base_drone";
        target.coordinate_frame = 8; // {MAV_FRAME_BODY_NED:8, MAV_FRAME_LOCAL_NED:1}
        target.type_mask = 1479; // velocity + yawrate
        target.position = geometry_msgs::Point();
        target.velocity = geometry_msgs::Vector3();
        target.acceleration_or_force = geometry_msgs::Vector3();
        target.yaw = 0.0;
        target.yaw_rate = 0.0;
    }

    void StateCallback(const mavros_msgs::State::ConstPtr& msg) {
        if (msg->mode == "OFFBOARD" && current_state.mode != "OFFBOARD") {
            ROS_INFO("Switched to OFFBOARD Mode!");
            offboard_start_time = ros::Time::now();
        }

        if (msg->mode != "OFFBOARD" && current_state.mode == "OFFBOARD") {
            ROS_INFO("Switched to %s Mode!", (msg->mode).c_str());
            InitializeTarget();
        }
        
        current_state = *msg;
    }
    
    void RCInCallback(const mavros_msgs::RCIn::ConstPtr& msg, int channel_index) {
        float cmd = rc_mapping(msg->channels[channel_index]);
        yaw_cmd = constrain_float(cmd, -1.0, 1.0);
    }

    void JoystickCallback(const mavros_msgs::ManualControl::ConstPtr& msg) {
        float cmd = msg->r;
        yaw_cmd = constrain_float(cmd, -1.0, 1.0);
    }

    void LocalPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        tf2::Quaternion q_tf;
        tf2::convert((msg->pose).orientation, q_tf);
        tf2::Matrix3x3 q_mat(q_tf);
        tf2Scalar yaw, pitch, roll;
        q_mat.getEulerYPR(yaw, pitch, roll);
        yaw_rad = wrap_2PI((float)yaw);
    }

private:
    ros::NodeHandle nh;
    ros::Publisher target_setpoint_pub;
    ros::Subscriber state_sub;
    ros::Subscriber local_pose_sub;
    ros::Subscriber rcin_sub;

    mavros_msgs::State current_state;
    mavros_msgs::PositionTarget target;
    
    float yaw_cmd; // in [-1.0, 1.0]
    float yaw_rad; // Down

    ros::Time offboard_start_time;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "forward_flight_node");
    ros::NodeHandle nh("~");

    ForwardCtrl ctrl;
    ros::Duration(1.0).sleep();
    ctrl.run();
}