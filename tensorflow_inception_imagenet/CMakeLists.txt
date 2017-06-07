cmake_minimum_required(VERSION 2.8.3)
project(tensorflow_inception_imagenet)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  vision_msgs
)

catkin_package(
 CATKIN_DEPENDS rospy sensor_msgs vision_msgs
)

catkin_install_python(
  PROGRAMS
    scripts/tensorflow_classifier.py
    scripts/tensorflow_listener.py
    scripts/node_lookup.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)