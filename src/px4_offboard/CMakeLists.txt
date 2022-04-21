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
    message_generation
)

find_package(Boost REQUIRED)

add_message_files(
  FILES
  Affordance.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES px4_offboard
  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# add_executable(offboard_test src/offboard_test.cpp)
# add_executable(set_attitude src/set_attitude.cpp)
add_executable(forward_flight 
  src/forward_flight.cpp
  src/rulebased_ctrl.cpp)

# target_link_libraries(offboard_test ${catkin_LIBRARIES})
# target_link_libraries(set_attitude ${catkin_LIBRARIES})
target_link_libraries(forward_flight ${catkin_LIBRARIES})