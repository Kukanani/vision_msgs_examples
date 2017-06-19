// Copyright 2015 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <rclcpp/rclcpp.hpp>

#include <memory>

#include "image_pipeline/camera_node.hpp"
#include "face_detector_node.hpp"
#include "image_pipeline/image_view_node.hpp"
#include "astra_camera/astra_driver.h"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::executors::SingleThreadedExecutor executor;

  // Connect the nodes as a pipeline: camera_node -> watermark_node -> image_view_node
  // std::shared_ptr<CameraNode> camera_node = nullptr;
  // try {
  //   camera_node = std::make_shared<CameraNode>("image", false);
  // } catch (const std::exception & e) {
  //   fprintf(stderr, "%s Exiting ..\n", e.what());
  //   return 1;
  // }

  // RGB
  size_t width = 1280;
  size_t height = 1024;
  double framerate = 30;

  // Depth
  size_t dwidth = 640;
  size_t dheight = 480;
  double dframerate = 30;
  astra_wrapper::PixelFormat dformat = astra_wrapper::PixelFormat::PIXEL_FORMAT_DEPTH_1_MM;

  rclcpp::node::Node::SharedPtr astra_node = rclcpp::node::Node::make_shared("astra_camera");
  rclcpp::node::Node::SharedPtr astra_private_node = rclcpp::node::Node::make_shared("astra_camera_");

  astra_wrapper::AstraDriver drv(astra_node, astra_private_node, width, height, framerate, dwidth, dheight, dframerate, dformat);


  auto face_detector_node =
    std::make_shared<FaceDetectorNode>("image", "image_with_faces", "face_detector_node");
  auto image_view_node = std::make_shared<ImageViewNode>("image_with_faces", "image_view_node", false);


  // executor.add_node(camera_node);
  executor.add_node(astra_node);
  executor.add_node(astra_private_node);
  executor.add_node(face_detector_node);
  executor.add_node(image_view_node);

  executor.spin();
  return 0;
}
