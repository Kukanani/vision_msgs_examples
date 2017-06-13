cmake_minimum_required(VERSION 2.8.3)
project(face_detector)

add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS roscpp cv_bridge sensor_msgs vision_msgs)
find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp cv_bridge std_msgs sensor_msgs vision_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} src/face_detector.cpp)
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBS})

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)