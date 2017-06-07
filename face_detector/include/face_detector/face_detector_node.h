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

#ifndef FACE_DETECTOR__FACE_DETECTOR_NODE_H__
#define FACE_DETECTOR__FACE_DETECTOR_NODE_H__

#include <memory>
#include <string>
#include <vector>

#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "sensor_msgs/Image.h"
#include "vision_msgs/Detection2D.h"
#include "vision_msgs/Detection2DArray.h"

// Node that receives an image, locates faces and draws boxes around them, and publishes it again.
class FaceDetectorNode
{
public:
  FaceDetectorNode(
    const std::string & input, const std::string & output,
    const std::string & node_name = "face_detector_node")
    : node_(node_name, "")
  {
    // Create a publisher on the input topic.
    image_pub_ = node_.advertise<sensor_msgs::Image>(output, 10);
    detections_pub_ = node_.advertise<vision_msgs::Detection2DArray>(output, 10);
    sub_ = node_.subscribe(input, 100, &FaceDetectorNode::cb_image, this);
    ros::spin();
  }

  void cb_image(const sensor_msgs::ImageConstPtr& msg)
  {
    // Create a cv::Mat from the image message (without copying).
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    cv::CascadeClassifier face_cascade;
    face_cascade.load("/home/adam/osrf/opencv/opencv/data/haarcascades/haarcascade_frontalface_default.xml");
    // Detect faces
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(cv_ptr->image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

    vision_msgs::Detection2DArray detection_array;
    detection_array.header.stamp = ros::Time::now();
    // Draw circles on the detected faces
    for( size_t i = 0; i < faces.size(); i++ )
    {
        cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        cv::ellipse(cv_ptr->image, center, cv::Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0 );

        vision_msgs::Detection2D detection;
        detection.bbox.x = faces[i].x;
        detection.bbox.y = faces[i].y;
        detection.bbox.width = faces[i].width;
        detection.bbox.height = faces[i].height;

        // normally we would also want to add ObjectHypothesisWithPose's, but
        // in this case, all we care about is the bounding box. The pose is
        // going to be at the center of the bounding box in this case, anyway.
        // However, by not using the ObjectHypothesis messages, we lose the
        // ability to define classes or have a distribution over categories, so
        // this behavior is not recommended for any but the most toy examples.
        detection_array.detections.push_back(detection);
    }
    detections_pub_.publish(detection_array);

    sensor_msgs::ImagePtr new_msg = cv_ptr->toImageMsg();
    // Publish it along.
    image_pub_.publish(*new_msg);
  }

private:
  ros::NodeHandle node_;

  ros::Subscriber sub_;
  ros::Publisher image_pub_;
  ros::Publisher detections_pub_;

  cv::VideoCapture cap_;
  cv::Mat frame_;
};

#endif
