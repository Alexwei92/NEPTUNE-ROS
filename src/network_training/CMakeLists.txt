cmake_minimum_required(VERSION 3.0.2)
project(network_training)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++14)

find_package(catkin REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES network_training
#  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)

include_directories(
## include
  ${catkin_INCLUDE_DIRS}
)