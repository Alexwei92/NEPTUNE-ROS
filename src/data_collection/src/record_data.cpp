#include <ros/ros.h>
#include <rosbag/bag.h>

#include <std_msgs/String.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/RCIn.h>
#include <mavros_msgs/State.h>
#include "data_collection/Custom.h"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>

#include <ctime>

#define LOOP_RATE_DEFAULT   10 // Hz


class GCS_Listener
{
public:
    GCS_Listener(std::string filename)
    {       
        try
        {
            my_bag.open(filename, rosbag::bagmode::Write);
            ROS_INFO("Logging file to %s", filename.c_str());
        }
        catch(const std::exception& e)
        {
            ROS_ERROR("%s", e.what());
            ros::shutdown();
            exit(1);
        }
        
        color_sub  = nh.subscribe("/d435i/color/image_raw", 10, &GCS_Listener::updateColor, this);
        depth_sub  = nh.subscribe("/d435i/aligned_depth_to_color/image_raw", 10, &GCS_Listener::updateDepth, this);

        // local_position_sub = nh.subscribe("/mavros/global_position/local", 10, &GCS_Listener::updateLocalPosition, this);
        // rel_alt_sub = nh.subscribe("/mavros/global_position/rel_alt", 10, &GCS_Listener::updateAlt, this);
        // vel_body_sub = nh.subscribe("/mavros/local_position/velocity_body", 10, &GCS_Listener::updateVelBody, this);
        // rc_in_sub = nh.subscribe("/mavros/rc/in", 10, &GCS_Listener::updateRCIn, this);
        // state_sub = nh.subscribe("/mavros/state", 10, &GCS_Listener::updateState, this);
    }

    void run()
    {
        ROS_INFO("Start Recording...");
        ros::Rate loop_rate(LOOP_RATE_DEFAULT);
        ros::Timer timer = nh.createTimer(ros::Duration(1.0/LOOP_RATE_DEFAULT), &GCS_Listener::iteration, this);
        ros::spin();

        my_bag.close();
        ROS_INFO("Stop Recording...");
    }


private:
    void iteration(const ros::TimerEvent&)
    {
        telemetry.header.seq += 1;
        telemetry.header.stamp = ros::Time::now();
        my_bag.write("/my_telemetry", ros::Time::now(), telemetry);

        color_image.header.seq = telemetry.header.seq;
        depth_image.header.seq = telemetry.header.seq;
        my_bag.write("/d435i/color/image_raw", ros::Time::now(), color_image);
        my_bag.write("/d435i/aligned_depth_to_color/image_raw", ros::Time::now(), depth_image);
    }

    void updateColor(const sensor_msgs::Image::ConstPtr& msg)
    {          
        color_image = *msg;
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, "bgr8"); ;
        cv::Mat frame = cv_ptr->image;
        cv::namedWindow("Image");
        cv::imshow("Image", frame);
        cv::waitKey(1);
    }

    void updateDepth(const sensor_msgs::Image::ConstPtr& msg)
    {          
        depth_image = *msg;

    }

    void updateLocalPosition(const nav_msgs::Odometry::ConstPtr& msg)
    {   
        telemetry.pose = msg->pose;
        telemetry.twist = msg->twist;
    }

    void updateAlt(const std_msgs::Float64::ConstPtr& msg)
    {
        telemetry.rel_alt = msg->data;
    }

    void updateVelBody(const geometry_msgs::TwistStamped::ConstPtr& msg)
    {
        telemetry.vel_twist = msg->twist;
    }

    void updateRCIn(const mavros_msgs::RCIn::ConstPtr& msg)
    {
        telemetry.channels = msg->channels;
    }

    void updateState(const mavros_msgs::State::ConstPtr& msg)
    {
        telemetry.mode = msg->mode;
    }

private:
    ros::NodeHandle nh;
    rosbag::Bag my_bag;
    ros::Subscriber color_sub;
    ros::Subscriber depth_sub;

    ros::Subscriber local_position_sub;
    ros::Subscriber rel_alt_sub;
    ros::Subscriber vel_body_sub;
    ros::Subscriber rc_in_sub;
    ros::Subscriber state_sub;

    sensor_msgs::Image color_image;
    sensor_msgs::Image depth_image;

    data_collection::Custom telemetry;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "GCS_Listener_node");
    ros::NodeHandle n("~");

    // bag file definition
    std::string location;
    n.getParam("location", location);

    std::time_t now = time(0);
    struct tm * timeinfo = localtime(&(now));
    char buffer [30];
    strftime(buffer,30,"%Y_%h_%d_%H_%M_%S.bag", timeinfo);
    
    ROS_INFO("Initializing...");
    GCS_Listener listener(location + '/' + buffer);
    ros::Duration(1.0).sleep();
    ROS_INFO("Ready!");
    listener.run();
}