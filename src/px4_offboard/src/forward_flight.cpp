#include <ros/ros.h>
#include <math.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/RCIn.h>
#include <mavros_msgs/ManualControl.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/SetMode.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <boost/bind.hpp>

#include "common.h"
#include "math_utils.h"
#include "rulebased_ctrl.h"
#include "px4_offboard/Affordance.h"


class ForwardCtrl
{
public:
    float LOOP_RATE_DEFAULT = 30; // Hz

public:
    ForwardCtrl()
    {
        InitializeTarget();
        
        ros::param::param<float>("~forward_speed", forward_speed, 0.5);
        ROS_INFO("Forward speed is set to %.2f m/s", forward_speed);

        ros::param::param<float>("~max_yawrate", max_yawrate, 45);
        ROS_INFO("Maximum yaw rate is set to %.2f deg/s", max_yawrate);

        // Publisher
        target_setpoint_pub = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);  

        // Subscriber
        // 1) state
        state_sub = nh.subscribe<mavros_msgs::State>("/mavros/state", 5, 
                &ForwardCtrl::StateCallback, this);
        // 2) local pose
        local_pose_sub = nh.subscribe<geometry_msgs::PoseStamped>("/mavros/local_position/pose", 5,
                &ForwardCtrl::LocalPoseCallback, this);
        // 3) rc / joystick / auto command
        #ifdef AUTO_MODE
        cmd_sub = nh.subscribe<std_msgs::Float32>("/my_controller/yaw_cmd", 5, 
                boost::bind(&ForwardCtrl::CmdCallback, this, _1));
        #else
            #ifdef SITL_MODE
            rc_sub = nh.subscribe<mavros_msgs::ManualControl>("/mavros/manual_control/control", 5, 
                    boost::bind(&ForwardCtrl::JoystickCallback, this, _1));
            #else
            rc_sub = nh.subscribe<mavros_msgs::RCIn>("/mavros/rc/in", 5, 
                    boost::bind(&ForwardCtrl::RCInCallback, this, _1, YAW_CHANNEL));
            #endif
        #endif

        // Affordance
        afford_sub = nh.subscribe<px4_offboard::Affordance>("/estimated_affordance", 5, 
                    &ForwardCtrl::AffordanceCallback, this);

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
                velocity_local.x = forward_speed * ratio;
                velocity_local.y = 0; 
                velocity_local.z = 0;
                if (target.coordinate_frame == 1) {
                    rotate_body_frame_to_NE(velocity_local.x, velocity_local.y, yaw_rad);
                }
                target.velocity = velocity_local;
                // yaw rate
                target.yaw_rate = -yaw_cmd * max_yawrate * DEG2RAD;

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

    void CmdCallback(const std_msgs::Float32::ConstPtr& msg) {
        float cmd = msg->data;
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

    void AffordanceCallback(const px4_offboard::Affordance::ConstPtr& msg) {
        if (msg != NULL) {
            bool in_bound = msg->in_bound;
            if (!in_bound) {
                set_mode("POSCTL");
            }
        }
    }

    void set_mode(std::string mode_name) {
        ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("mavros/set_mode");
        mavros_msgs::SetMode new_mode;
        new_mode.request.custom_mode = mode_name;
        while (!new_mode.response.mode_sent) {
            set_mode_client.call(new_mode);
        }
    }

private:
    ros::NodeHandle nh;
    ros::Publisher target_setpoint_pub;
    ros::Subscriber state_sub;
    ros::Subscriber local_pose_sub;
    ros::Subscriber rc_sub;
    ros::Subscriber cmd_sub;
    ros::Subscriber afford_sub;

    mavros_msgs::State current_state;
    mavros_msgs::PositionTarget target;
    
    float forward_speed; // m/s
    float max_yawrate; // deg/s

    float yaw_cmd; // in [-1.0, 1.0]
    float yaw_rad; // Down positive

    ros::Time offboard_start_time;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "forward_flight_node");
    ros::NodeHandle nh("~");

    ForwardCtrl ctrl;
    ros::Time tic = ros::Time::now();
    while (ros::Time::now() - tic < ros::Duration(1.0)) {
        // sleep for one second
    }

    ctrl.run();

    return 0;
}