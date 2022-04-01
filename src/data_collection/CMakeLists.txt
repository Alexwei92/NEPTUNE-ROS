cmake_minimum_required(VERSION 2.8.3)
project(data_collection)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++11)
add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS 
    roscpp
    rosbag
    std_msgs
    sensor_msgs
    mavros_msgs
    geometry_msgs
    cv_bridge
    image_transport
)

find_package(OpenCV 3 REQUIRED)

# add_message_files(
#   FILES
#   Custom.msg
# )

# generate_messages(
#   DEPENDENCIES
#   std_msgs
#   geometry_msgs
# )

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES neptune
#  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)

include_directories(
## include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(record_data src/record_data.cpp)

target_link_libraries(record_data ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
