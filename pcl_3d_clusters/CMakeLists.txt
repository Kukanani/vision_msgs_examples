cmake_minimum_required(VERSION 2.8.3)
project(pcl_3d_clusters)

add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS pcl_ros tf sensor_msgs vision_msgs)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS pcl_ros tf sensor_msgs vision_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(point_cloud_processor src/point_cloud_processor.cpp)
add_dependencies(point_cloud_processor ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(point_cloud_processor ${catkin_LIBRARIES})