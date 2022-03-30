cmake_minimum_required(VERSION 2.8.3)
project(neptune_ros)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++11)
add_compile_options(-std=c++14)

find_package(catkin REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES neptune
#  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)

include_directories(
## include
  ${catkin_INCLUDE_DIRS}
)