cmake_minimum_required(VERSION 2.8.3)
project(px4_offboard)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++11)
add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS 
    roscpp
    rospy
    rosbag
    std_msgs
    sensor_msgs
    mavros_msgs
    geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES px4_offboard
#  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(px4_offboard src/offboard.cpp)
add_executable(fake_yawCmd src/fake_yawCmd.cpp)

target_link_libraries(px4_offboard ${catkin_LIBRARIES})
target_link_libraries(fake_yawCmd ${catkin_LIBRARIES})