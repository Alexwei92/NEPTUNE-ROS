#include <ros/ros.h>
#include <rosbag/bag.h>

#include <std_msgs/String.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
// #include <theora_image_transport/Packet.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/RCIn.h>
#include <mavros_msgs/State.h>
#include <neptune_ros/Custom.h>

#include <ctime>

#define LOOP_RATE_DEFAULT   10.0


/** A CGS_Listener Class **/
class GCS_Listener
{
    public:
    /**
     *  Initialize
    **/
    GCS_Listener(std::string filename,
             double loop_rate,
             std::string color_image_topic,
             std::string depth_image_topic)
    {       
        try
        {
            _bag.open(filename, rosbag::bagmode::Write);
            ROS_INFO("Logging file to %s", filename.c_str());
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            ros::shutdown();
            exit(1);
        }
        
        if (loop_rate > 0.0)
        { 
            _loop_rate = loop_rate;
        }
        else
        {
            _loop_rate = LOOP_RATE_DEFAULT;
        }
        

        _color_sub  = _n.subscribe(color_image_topic, 10, &GCS_Listener::updateColor, this);
        _depth_sub  = _n.subscribe(depth_image_topic, 10, &GCS_Listener::updateDepth, this);

        _local_position_sub = _n.subscribe("/mavros/global_position/local", 10, &GCS_Listener::updateLocalPosition, this);
        _rel_alt_sub = _n.subscribe("/mavros/global_position/rel_alt", 10, &GCS_Listener::updateAlt, this);
        _vel_body_sub = _n.subscribe("/mavros/local_position/velocity_body", 10, &GCS_Listener::updateVelBody, this);
        _rc_in_sub = _n.subscribe("/mavros/rc/in", 10, &GCS_Listener::updateRCIn, this);
        _state_sub = _n.subscribe("/mavros/state", 10, &GCS_Listener::updateState, this);
    }


    /** 
     * Main run function
    **/
    void run()
    {
        ros::Timer timer = _n.createTimer(ros::Duration(1.0/_loop_rate), &GCS_Listener::iteration, this);
        ros::spin();

        _bag.close();
    }


    private:

    void iteration(const ros::TimerEvent&)
    {
        _telemetry.header.seq += 1;
        _telemetry.header.stamp = ros::Time::now();
        _bag.write("/my_telemetry", ros::Time::now(), _telemetry);

        _color_image.header.seq = _telemetry.header.seq;
        _depth_image.header.seq = _telemetry.header.seq;
        _bag.write("/d435i/color/image_raw", ros::Time::now(), _color_image);
        _bag.write("/d435i/aligned_depth_to_color/image_raw", ros::Time::now(), _depth_image);


    }

    // Color Image
    void updateColor(const sensor_msgs::Image::ConstPtr& msg)
    {          
        _color_image = *msg;
    }

    // Depth Image
    void updateDepth(const sensor_msgs::Image::ConstPtr& msg)
    {          
        _depth_image = *msg;
    }

    // Local Position
    void updateLocalPosition(const nav_msgs::Odometry::ConstPtr& msg)
    {   
        _telemetry.pose = msg->pose;
        _telemetry.twist = msg->twist;

    }

    // Relative Altitude
    void updateAlt(const std_msgs::Float64::ConstPtr& msg)
    {
        _telemetry.rel_alt = msg->data;

    }

    // Body Velocity
    void updateVelBody(const geometry_msgs::TwistStamped::ConstPtr& msg)
    {
        _telemetry.vel_twist = msg->twist;
    }

    // RC Input
    void updateRCIn(const mavros_msgs::RCIn::ConstPtr& msg)
    {
        _telemetry.channels = msg->channels;
    }

    // State
    void updateState(const mavros_msgs::State::ConstPtr& msg)
    {
        _telemetry.mode = msg->mode;
    }


    private:

    double _loop_rate;

    ros::NodeHandle _n;
    rosbag::Bag _bag;
    ros::Subscriber _color_sub;
    ros::Subscriber _depth_sub;

    ros::Subscriber _local_position_sub;
    ros::Subscriber _rel_alt_sub;
    ros::Subscriber _vel_body_sub;
    ros::Subscriber _rc_in_sub;
    ros::Subscriber _state_sub;

    sensor_msgs::Image _color_image;
    sensor_msgs::Image _depth_image;

    neptune_ros::Custom _telemetry;

};


int main(int argc, char **argv)
{
    // initialize ros node
    ros::init(argc, argv, "GCS_Listener_node");
    ros::NodeHandle n("~");

    // loop rate
    double rate;
    n.getParam("rate", rate);
    ros::Rate loop_rate(rate);

    // bag file definition
    std::string location;
    n.getParam("location", location);

    std::time_t now = time(0);
    struct tm * timeinfo = localtime(&(now));
    char buffer [30];
    strftime(buffer,30,"%Y_%h_%d_%H_%M_%S.bag", timeinfo);

    // image topics
    std::string color_image_topic;
    std::string depth_image_topic;
    n.getParam("color_image_topic", color_image_topic);
    n.getParam("depth_image_topic", depth_image_topic);
    
    // init class
    GCS_Listener listener(location + '/' + buffer,
                      rate,
                      color_image_topic,
                      depth_image_topic);

    ros::Duration(1.0).sleep(); // let Listener update its internal states
    listener.run();

}
