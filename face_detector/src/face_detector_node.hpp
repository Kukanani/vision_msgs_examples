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

#ifndef FACE_DETECTOR__FACE_DETECTOR_NODE_HPP_
#define FACE_DETECTOR__FACE_DETECTOR_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "intra_process_demo/image_pipeline/common.hpp"

// Node that receives an image, locates faces and draws boxes around them, and publishes it again.
class FaceDetectorNode : public rclcpp::Node
{
public:
  FaceDetectorNode(
    const std::string & input, const std::string & output,
    const std::string & node_name = "face_detector_node")
  : Node(node_name, "", true)
  {
    auto qos = rmw_qos_profile_sensor_data;
    // Create a publisher on the input topic.
    pub_ = this->create_publisher<sensor_msgs::msg::Image>(output, qos);
    std::weak_ptr<std::remove_pointer<decltype(pub_.get())>::type> captured_pub = pub_;
    // Create a subscription on the output topic.
    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      input, [captured_pub](sensor_msgs::msg::Image::UniquePtr msg) {
      std::cout << "image received!" << std::endl;
      auto pub_ptr = captured_pub.lock();
      if (!pub_ptr) {
        return;
      }
      // Create a cv::Mat from the image message (without copying).
      cv::Mat cv_mat(
        msg->height, msg->width,
        encoding2mat_type(msg->encoding),
        msg->data.data());
      cv::cvtColor(cv_mat, cv_mat, CV_BGR2RGB);

      cv::CascadeClassifier face_cascade;
      face_cascade.load("/home/adam/osrf/opencv/opencv/data/haarcascades/haarcascade_frontalface_default.xml");
      // Detect faces
      std::vector<cv::Rect> faces;
      face_cascade.detectMultiScale( cv_mat, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

      // Draw circles on the detected faces
      for( size_t i = 0; i < faces.size(); i++ )
      {
          cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
          cv::ellipse( cv_mat, center, cv::Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0 );
      }
      // Publish it along.
      pub_ptr->publish(msg);
    }, qos);
  }

private:
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;

  cv::VideoCapture cap_;
  cv::Mat frame_;
};

#endif
