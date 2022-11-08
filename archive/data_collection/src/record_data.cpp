#include <ros/ros.h>
#include <rosbag/bag.h>

#include <std_msgs/String.h>
#include <std_msgs/UInt16.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/RCIn.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>

#include <ctime>

#define LOOP_RATE_DEFAULT   10 // Hz

void resize_image(const sensor_msgs::Image::ConstPtr& msg, 
        sensor_msgs::Image& dst, cv::Size new_size, std::string encoding)
{
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, encoding); ;
    cv::Mat frame = cv_ptr->image;
    cv::Mat frame_resize;
    cv::resize(frame, frame_resize, new_size);       
    sensor_msgs::ImagePtr compImg = cv_bridge::CvImage(std_msgs::Header(), encoding, frame_resize).toImageMsg();;
    dst = *compImg; 
}

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
        
        // Subscriber
        color_sub  = nh.subscribe<sensor_msgs::Image>(
                "/d435i/color/image_raw", 5, &GCS_Listener::ColorCallback, this);
        depth_sub  = nh.subscribe<sensor_msgs::Image>(
                "/d435i/aligned_depth_to_color/image_raw",5, &GCS_Listener::DepthCallback, this);

        rcin_sub = nh.subscribe<mavros_msgs::RCIn>(
                "/mavros/rc/in", 5, &GCS_Listener::RCInCallback, this);
        local_pose_sub = nh.subscribe<geometry_msgs::PoseStamped>(
                "/mavros/local_position/pose", 5, &GCS_Listener::LocalPoseCallback, this);
        global_loc_sub = nh.subscribe<sensor_msgs::NavSatFix>(
                "/mavros/global_position/global", 5, &GCS_Listener::GlobalLocationCallback, this);
        compass_sub = nh.subscribe<std_msgs::Float64>(
                "/mavros/global_position/compass_hdg", 5, &GCS_Listener::CompassCallback, this);
        vel_body_sub = nh.subscribe<geometry_msgs::TwistStamped>(
                "/mavros/local_position/velocity_body", 5, &GCS_Listener::VelBodyCallback, this);
    }

    void run()
    {
        ros::Rate loop_rate(LOOP_RATE_DEFAULT);
        int count = 0;
        ROS_INFO("Start Recording...");
        while(ros::ok()) {
            ros::Time time = ros::Time::now();

            my_bag.write("/my_telemetry/rc/in", time, yaw_cmd);
            my_bag.write("/my_telemetry/local_pose", time, local_pose);
            my_bag.write("/my_telemetry/global_location", time, global_location);
            my_bag.write("/my_telemetry/velocity_body", time, vel_body);

            color_image.header.seq = count;
            depth_image.header.seq = count;
            my_bag.write("/d435i/color/image_raw", time, color_image);
            my_bag.write("/d435i/aligned_depth_to_color/image_raw", time, depth_image);

            ros::spinOnce();
            loop_rate.sleep();
            count++;
        }
    
        my_bag.close();
        ROS_INFO("Stop Recording...");
    }


private:
    void ColorCallback(const sensor_msgs::Image::ConstPtr& msg)
    {          
        resize_image(msg, color_image, cv::Size(300,300), "bgr8");
    }

    void DepthCallback(const sensor_msgs::Image::ConstPtr& msg)
    {          
        resize_image(msg, depth_image, cv::Size(300,300), "mono16");
    }

    void RCInCallback(const mavros_msgs::RCIn::ConstPtr& msg)
    {
        yaw_cmd.data = msg->channels[3];
    }

    void LocalPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        local_pose = msg->pose;
    }

    void GlobalLocationCallback(const sensor_msgs::NavSatFix::ConstPtr& msg) 
    {
        global_location.x = msg->latitude;
        global_location.y = msg->longitude;
    }

    void CompassCallback(const std_msgs::Float64::ConstPtr& msg) 
    {
        global_location.z = msg->data;
    }

    void VelBodyCallback(const geometry_msgs::TwistStamped::ConstPtr& msg)
    {
        vel_body = msg->twist;
    }

private:
    ros::NodeHandle nh;
    rosbag::Bag my_bag;
    ros::Subscriber color_sub;
    ros::Subscriber depth_sub;

    ros::Subscriber rcin_sub;
    ros::Subscriber local_pose_sub;
    ros::Subscriber global_loc_sub;
    ros::Subscriber compass_sub;
    ros::Subscriber vel_body_sub;

    sensor_msgs::Image color_image;
    sensor_msgs::Image depth_image;
    std_msgs::UInt16 yaw_cmd;
    geometry_msgs::Pose local_pose;
    geometry_msgs::Point global_location;
    geometry_msgs::Twist vel_body;
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