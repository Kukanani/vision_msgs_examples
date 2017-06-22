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

#include "face_detector_node.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  std::shared_ptr<FaceDetectorNode> face_detector_node = nullptr;
  try {
    face_detector_node = std::make_shared<FaceDetectorNode>(
        "image", "image_with_faces", "face_detector_node", false);
  } catch (const std::exception & e) {
    fprintf(stderr, "%s Exiting..\n", e.what());
    return 1;
  }

  rclcpp::spin(face_detector_node);
  return 0;
}
