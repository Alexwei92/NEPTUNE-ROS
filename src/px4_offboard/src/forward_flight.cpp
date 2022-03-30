#include <ros/ros.h>
#include <boost/bind.hpp>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/Header.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/RCIn.h>
#include <mavros_msgs/State.h>

#include "common.h"
#include "math_utils.h"

#define LOOP_RATE_DEFAULT   10


float linear_mapping(uint16_t value) {
    float result = ((float)(value) - CH_MIN) / (CH_MAX - CH_MIN); 
    result = result * 2.0 - 1.0;
    return result;
}

class ForwardCtrl
{
public:
    ForwardCtrl()
    {
        target.header = std_msgs::Header();
        target.header.frame_id = "base_drone";
        
        // Publisher
        target_setpoint_pub = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);  

        // Subscriber
        rcin_sub = nh.subscribe<mavros_msgs::RCIn>("/mavros/rc/in", 
                10, boost::bind(&ForwardCtrl::RCInCallback, this, _1, YAW_CHANNEL));
    }

    void run()
    {
        ros::Rate loop_rate(LOOP_RATE_DEFAULT);
        ROS_INFO("Start Offboard Mode!");
        while (ros::ok()) {
            target.header.stamp = ros::Time::now();
            target.header.seq++;
            // bitmask
            target.coordinate_frame = 1; // MAV_FRAME_LOCAL_NED
            target.type_mask = 4; //ignore yaw
            // velocity
            geometry_msgs::Vector3 velocity;
            velocity.x = 1;
            velocity.y = 0;
            velocity.z = 0;
            target.velocity = velocity;
            // yaw rate
            target.yaw_rate = yaw_cmd * MAX_YAWRATE * DEG2RAD;
            
            target_setpoint_pub.publish(target);
            ros::spinOnce();
            loop_rate.sleep();
        }
        ROS_INFO("Stop Offboard Mode!");
    }

private:
    void RCInCallback(const mavros_msgs::RCIn::ConstPtr& msg, int channel_index) {
        float cmd = linear_mapping(msg->channels[channel_index]);
        yaw_cmd = constrain_float(cmd, -1.0, 1.0);
    }

private:
    ros::NodeHandle nh;
    ros::Publisher target_setpoint_pub;
    ros::Subscriber rcin_sub;

    mavros_msgs::PositionTarget target;
    float yaw_cmd;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "forward_flight_node");
    ros::NodeHandle nh("~");

    ForwardCtrl ctrl;
    ros::Duration(1.0).sleep();
    ctrl.run();
}