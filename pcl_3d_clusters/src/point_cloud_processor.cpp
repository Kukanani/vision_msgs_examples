// Copyright (c) 2015, Adam Allevato
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "pcl_3d_clusters/point_cloud_processor.h"

#include "vision_msgs/Detection3D.h"
#include "vision_msgs/Detection3DArray.h"
#include "vision_msgs/ObjectHypothesis.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "point_cloud_processor");
  PointCloudProcessor pcu;
  ros::spin();
  return 0;
}

PointCloudProcessor::PointCloudProcessor() :
  node("point_cloud_processor")
{
  ros::NodeHandle privateNode("~");
  if(!privateNode.getParam("input_cloud_topic", cloudTopicName)) {
    cloudTopicName = "/camera/depth/points";
    ROS_INFO_STREAM("listening for point clouds on topic " << cloudTopicName << ", change by setting the input_cloud_topic param");
  }

  voxelPublisher = node.advertise<sensor_msgs::PointCloud2>("/voxel_scene",1);
  allPlanesPublisher = node.advertise<sensor_msgs::PointCloud2>("/all_planes",1);
  largestObjectPublisher = node.advertise<sensor_msgs::PointCloud2>("/largest_object",1);
  allObjectsPublisher = node.advertise<sensor_msgs::PointCloud2>("/all_objects",1);
  detectionPublisher = node.advertise<vision_msgs::Detection3DArray>("/detection_array",1);

  pointCloudSub = node.subscribe(cloudTopicName, 1, &PointCloudProcessor::cb_process, this);
}

bool compareClusterSize(const PCP& a, const PCP& b) { return a->points.size() > b->points.size(); }

void PointCloudProcessor::cb_process(sensor_msgs::PointCloud2ConstPtr inputMessage) {
  if(inputMessage->height * inputMessage->width < 3) {
    ROS_DEBUG("Not segmenting cloud, it's too small.");
    return;
  }

  inputCloud = PCP(new PC());
  processCloud = PCP(new PC());
  inputCloud->points.clear();
  originalCloudFrame = inputMessage->header.frame_id;

  pcl::fromROSMsg(*inputMessage, *inputCloud);
  sensor_msgs::PointCloud2 transformedMessage;

  if(inputCloud->points.size() <= minClusterSize) {
    ROS_INFO_STREAM("point cloud is too small to segment: Min: " << minClusterSize << ", actual: " << inputCloud->points.size());
    return;
  }

  //clip
  int preVoxel = inputCloud->points.size();

  //TODO:
  //   - "classify" based on cluster size
  //   - use PCA to find principal directions and calculate the 6D pose
  //   - compose and publish Detection3D message.

  *inputCloud = *(clipByDistance(inputCloud, -10, 10, -10, 10, 0.01, 2));

  *inputCloud = *(voxelGridify(inputCloud, voxelLeafSize));

  if(!inputCloud->points.empty() && inputCloud->points.size() < preVoxel) {
    // Publish voxelized
    if(_publishVoxelScene) {
      sensor_msgs::PointCloud2 voxelized_cloud;
      pcl::toROSMsg(*inputCloud, voxelized_cloud);
      voxelized_cloud.header.frame_id = originalCloudFrame;
      voxelPublisher.publish(voxelized_cloud);
    }

    //remove planes
    inputCloud = removePrimaryPlanes(inputCloud, maxPlaneSegmentationIterations, segmentationDistanceThreshold, percentageToAnalyze);

    if(_publishAllObjects) {
      pcl::toROSMsg(*inputCloud, transformedMessage);
      transformedMessage.header.frame_id = originalCloudFrame;
      allObjectsPublisher.publish(transformedMessage);
    }
    auto clusters = cluster(inputCloud, clusterTolerance, minClusterSize, maxClusterSize);

    vision_msgs::Detection3DArray detection_arr;
    for (auto cluster : clusters) {
      vision_msgs::Detection3D detection;

      std::vector<float> min = {1000, 1000, 1000};
      std::vector<float> max = {-1000, -1000, -1000};

      for(auto point : cluster->points) {
        min[0] = std::min(point.x, min[0]);
        min[1] = std::min(point.y, min[1]);
        min[2] = std::min(point.z, min[2]);

        max[0] = std::max(point.x, max[0]);
        max[1] = std::max(point.y, max[1]);
        max[2] = std::max(point.z, max[2]);
      }

      float size_x = max[0] - min[0];
      float size_y = max[1] - min[1];
      float size_z = max[2] - min[2];

      float max_size = std::max(std::max(size_x, size_y), size_z);
      float min_size = std::min(std::min(size_x, size_y), size_z);

      int shape_class = 0;
      if(max_size > 3 * min_size) {
        shape_class = 1;
      }
      int size_class = 2;
      if(max_size > 0.1) {
        size_class = 3;
      } if(max_size > 0.3) {
        size_class = 4;
      }

      Eigen::Vector4f clusterCentroid;
      pcl::compute3DCentroid(*cluster, clusterCentroid);

      for(int j=0; j < 5; ++j) {
        vision_msgs::ObjectHypothesisWithPose result;
        result.pose.position.x = clusterCentroid[0];
        result.pose.position.y = clusterCentroid[1];
        result.pose.position.z = clusterCentroid[2];
        result.pose.orientation.w = 1.0; // default orientation
        detection.results.push_back(result);
      }

      for (int i=0; i<=1; ++i) {
        float score = (i == shape_class ? 0.9 : 0.1);
        detection.results[i].score = score;
        detection.results[i].id = i;
      }
      for (int i=2; i<=4; ++i) {
        float score = (i == size_class ? 0.8 : 0.1);
        detection.results[i].score = score;
        detection.results[i].id = i;
      }

      // we could also fill the source_cloud variable here, but it's optional.

      detection_arr.detections.push_back(detection);
    }
    detection_arr.header = std_msgs::Header();
    detection_arr.header.frame_id = originalCloudFrame;
    detectionPublisher.publish(detection_arr);

  } else {
    if(inputCloud->points.empty()) {
      ROS_WARN_STREAM("After filtering, the cloud contained no points. No segmentation will occur.");
    }
    else {
      ROS_ERROR_STREAM("After filtering, the cloud contained " << inputCloud->points.size() << " points. This is more than BEFORE the voxel filter was applied, so something is wrong. No segmentation will occur.");
    }
  }
  return;
}

PCP& PointCloudProcessor::clipByDistance(PCP &unclipped,
    float minX, float maxX, float minY, float maxY, float minZ, float maxZ) {

  processCloud->resize(0);

  // We must build a condition.
  // And "And" condition requires all tests to check true. "Or" conditions also available.
  // Checks available: GT, GE, LT, LE, EQ.
  pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr clip_condition(new pcl::ConditionAnd<pcl::PointXYZRGB>);
  clip_condition->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
    new pcl::FieldComparison<pcl::PointXYZRGB>("x", pcl::ComparisonOps::GT, minX)));
  clip_condition->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
    new pcl::FieldComparison<pcl::PointXYZRGB>("x", pcl::ComparisonOps::LT, maxX)));
  clip_condition->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
    new pcl::FieldComparison<pcl::PointXYZRGB>("y", pcl::ComparisonOps::GT, minY)));
  clip_condition->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
    new pcl::FieldComparison<pcl::PointXYZRGB>("y", pcl::ComparisonOps::LT, maxY)));
  clip_condition->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
    new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::GT, minZ)));
  clip_condition->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
    new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::LT, maxZ)));

  // Filter object.
  pcl::ConditionalRemoval<pcl::PointXYZRGB> filter;
  filter.setCondition(clip_condition);
  filter.setInputCloud(unclipped);
  // If true, points that do not pass the filter will be set to a certain value (default NaN).
  // If false, they will be just removed, but that could break the structure of the cloud
  // (organized clouds are clouds taken from camera-like sensors that return a matrix-like image).
  filter.setKeepOrganized(true);
  // If keep organized was set true, points that failed the test will have their Z value set to this.
  filter.setUserFilterValue(0.0);

  filter.filter(*processCloud);
  return processCloud;
}

PCP& PointCloudProcessor::voxelGridify(PCP &loose, float gridSize) {
  //ROS_INFO("Voxel grid filtering...");

  processCloud->resize(0);
  // Create the filtering object: downsample the dataset
  pcl::VoxelGrid<PointType> vg;
  vg.setInputCloud(loose);
  vg.setLeafSize(gridSize, gridSize, gridSize);
  vg.filter(*processCloud);

  return processCloud;
}

PCP& PointCloudProcessor::removePrimaryPlanes(PCP &input, int maxIterations, float thresholdDistance,
    float percentageGood) {
  //ROS_INFO("Filtering planes...");
  PCP planes(new PC());
  PCP planeCloud(new pcl::PointCloud<PointType> ());

  processCloud->resize(0);
  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<PointType> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (maxIterations);
  seg.setDistanceThreshold (thresholdDistance);

  pcl::PointIndices::Ptr planeIndices(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PCDWriter writer;

  //how many points to get leave
  int targetSize = percentageGood * input->points.size();
  //ROS_INFO("target size: %d", targetSize);

  while(input->points.size() > targetSize) {
    seg.setInputCloud(input);
    seg.segment (*planeIndices, *coefficients);

    if(planeIndices->indices.size () == 0) {
      ROS_ERROR("Could not find any good planes in the point cloud.");
      break;
    }
    // Segment the largest planar component from the remaining cloud
    pcl::ExtractIndices<PointType> extract;
    extract.setInputCloud(input);
    extract.setIndices(planeIndices);

    //extract.setNegative(false);
    extract.filter (*planeCloud);
    //store it for the planes message
    planes->insert(planes->end(), planeCloud->begin(), planeCloud->end());

    //now actually take it out
    extract.setNegative(true);
    extract.filter(*processCloud);
    input = processCloud;
    //ROS_INFO("removed a plane.");
  }

  // Publish dominant planes
  if(_publishAllPlanes) {
    sensor_msgs::PointCloud2 planes_pc2;
    pcl::toROSMsg(*planes, planes_pc2);
    planes_pc2.header.frame_id = originalCloudFrame;
    allPlanesPublisher.publish(planes_pc2);
  }

  return input;
}

std::vector<PCP> PointCloudProcessor::cluster(PCP &input, float clusterTolerance,
    int minClusterSize, int maxClusterSize) {
  clusters.clear();

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
  tree->setInputCloud (input);

  IndexVector cluster_indices;
  pcl::EuclideanClusterExtraction<PointType> ec;
  ec.setInputCloud(input);

  ec.setClusterTolerance(clusterTolerance);
  ec.setMinClusterSize(minClusterSize);
  ec.setMaxClusterSize(maxClusterSize);
  ec.setSearchMethod(tree);

  ec.extract (cluster_indices);

  if(cluster_indices.empty()) return clusters;

  // go through the set of indices. Each set of indices is one cloud
  for (IndexVector::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
    //extract all the points based on the set of indices
    processCloud = PCP(new PC());
    for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end (); ++pit) {
      processCloud->points.push_back (input->points[*pit]);
    }
    processCloud->width = processCloud->points.size();
    processCloud->height = 1;
    processCloud->is_dense = true;

    clusters.push_back(processCloud);
  }

  if(clusters.size() > 0) {
    std::sort(clusters.begin(), clusters.end(), compareClusterSize);
  }

  return clusters;
}