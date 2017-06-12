cmake_minimum_required(VERSION 2.8.3)
project(tensorflow_inception_imagenet)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  vision_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES tensorflow_inception_imagenet
 CATKIN_DEPENDS rospy sensor_msgs vision_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)