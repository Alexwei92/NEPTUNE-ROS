#include <ros/ros.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Quaternion.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/RCIn.h>
#include <mavros_msgs/State.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include "math_utils.h"

#define LOOP_RATE_DEFAULT   10
#define CH_MIN 982
#define CH_MAX 2006
#define YAW_CHANNEL 3

/*
*/
float linear_mapping(uint16_t value) {
    float result = ((float)(value) - CH_MIN) / (CH_MAX - CH_MIN); 
    // result = result * 2.0 - 1.0;
    return result;
}

/*
*/
class Offboard_Ctrl
{
public:
    Offboard_Ctrl()
    {
        att.header = std_msgs::Header();
        att.header.frame_id = "base_footprint";
        
        // Publisher
        att_setpoint_pub = nh.advertise<mavros_msgs::AttitudeTarget>("/mavros/setpoint_raw/attitude", 1);  

        // Subscriber
        rcin_sub = nh.subscribe("/mavros/rc/in", 10, &Offboard_Ctrl::RCInCallback, this);
    }

    void run()
    {
        ros::Rate loop_rate(10);
        ROS_INFO("Start Offboard Mode!");
        while (ros::ok()) {
            att.header.stamp = ros::Time::now();
            att.header.seq++;

            // geometry_msgs::Vector3 body_rate;
            // body_rate.x = yaw_cmd * 45.0 / 180.0 * 3.14159f;
            // att.body_rate = body_rate;
            
            // tf2::Quaternion quat_tf;
            // quat_tf.setRPY( 0, yaw_cmd, 0);
            // att.orientation = tf2::toMsg(quat_tf);
            // att.orientation = geometry_msgs::Quaternion();
            
            att.thrust = yaw_cmd;
            // att.type_mask = 7; //ignore body rate

            att_setpoint_pub.publish(att);
            ros::spinOnce();
            loop_rate.sleep();
        }
        ROS_INFO("Stop Offboard Mode!");
    }

private:
    void RCInCallback(const mavros_msgs::RCIn::ConstPtr& msg) {
        float cmd = linear_mapping(msg->channels[YAW_CHANNEL]);
        yaw_cmd = constrain_float(cmd, -1.0, 1.0);
    }

private:
    ros::NodeHandle nh;
    ros::Publisher att_setpoint_pub;
    ros::Subscriber rcin_sub;

    mavros_msgs::AttitudeTarget att;
    float yaw_cmd;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_node");
    ros::NodeHandle nh("~");

    Offboard_Ctrl tmp = Offboard_Ctrl();
    ros::Duration(1.0).sleep();
    tmp.run();
}