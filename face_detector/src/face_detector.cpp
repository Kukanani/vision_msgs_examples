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

#ifndef FACE_DETECTOR__FACE_DETECTOR_CPP_
#define FACE_DETECTOR__FACE_DETECTOR_CPP_

#include <memory>
#include <string>
#include <vector>

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "sensor_msgs/Image.h"
#include "vision_msgs/Detection2DArray.h"
#include "vision_msgs/Detection2D.h"
#include "vision_msgs/ObjectHypothesisWithPose.h"

// Node that receives an image, locates faces and draws boxes around them, and publishes it again.
class FaceDetector
{
public:
  FaceDetector() : face_cascade_(), nh_()
  {
    image_sub_ = nh_.subscribe("/camera/image_raw", 1,
      &FaceDetector::cb_image, this);
    image_pub_ = nh_.advertise<vision_msgs::Detection2DArray>("face_detector/image_with_faces", 1);
    detections_pub_ = nh_.advertise<sensor_msgs::Image>("face_detector/detections", 1);
    face_cascade_.load("haarcascade_frontalface_default.xml");
  }

  void cb_image(const sensor_msgs::ImageConstPtr& msg)
  {
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

    // Detect faces

    std::vector<cv::Rect> faces;
    face_cascade_.detectMultiScale(cv_ptr->image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

    vision_msgs::Detection2DArray detections;
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

        vision_msgs::ObjectHypothesisWithPose hypo;
        hypo.id = 0;
        hypo.score = 1;
        detection.results.push_back(hypo);
        detections.detections.push_back(detection);
    }
    // Publish it along.
    image_pub_.publish(cv_ptr->toImageMsg());
    detections_pub_.publish(detections);
  }

private:
  ros::Publisher image_pub_;
  ros::Subscriber image_sub_;
  ros::Publisher detections_pub_;
  ros::NodeHandle nh_;
  cv::CascadeClassifier face_cascade_;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "face_detector");

  FaceDetector fd;

  ros::spin();
  return 0;
}

#endif
