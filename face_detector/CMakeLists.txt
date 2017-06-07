cmake_minimum_required(VERSION 2.8.3)
project(face_detector)
add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  std_msgs
  sensor_msgs
  vision_msgs
)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES face_detection_bridge
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs vision_msgs
#  DEPENDS system_lib
)

add_executable(face_detector src/face_detector_node.cpp)
add_dependencies(face_detector vision_msgs)
target_link_libraries(face_detector ${catkin_LIBRARIES} ${OpenCV_LIBS})