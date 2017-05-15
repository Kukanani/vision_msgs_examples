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

#ifndef _POINT_CLOUD_UTILS_H_
#define _POINT_CLOUD_UTILS_H_

#define _USE_MATH_DEFINES
#include <cmath>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/common/transforms.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

//useful PCL typedefs
typedef pcl::PointXYZRGB PointType;

typedef pcl::PointCloud<PointType> PC;
typedef pcl::PointCloud<PointType>::Ptr PCP;
typedef std::vector<pcl::PointIndices> IndexVector;

class PointCloudUtils {
private:

  /*================================================*/
  /* CLASS VARS */
  /*================================================*/
  /// Standard ROS node handle
  ros::NodeHandle node;

  /// Publishes planes cut from the scene
  ros::Publisher allPlanesPublisher;
  /// Publishes clipped cloud after voxel gridification
  ros::Publisher voxelPublisher;
  /**
   * Publishes the various clusters identified by the Euclidean algorithm.
   * All of these will lie within the bounded_scene point cloud.
   * NOTE: concatenated clouds from pre_clustering = bounded_scene - planes
   */
  ros::Publisher allObjectsPublisher;

  /// Publishes the first (largest) cluster in the scene.
  ros::Publisher largestObjectPublisher;

  /// Publishes detected objects in vision_msgs format.
  ros::Publisher detectionPublisher;

  /// Listens for new point clouds to process
  ros::Subscriber pointCloudSub;

  /*================================================*/
  /* SEGMENTATION PARAMS */
  /*================================================*/
  //The minimum camera-space X for the working area bounding box

  ///flags for publishing various intermediate point clouds
  bool _publishAllObjects = true;

  bool _publishAllPlanes = false;
  bool _publishLargestObject = false;
  bool _publishVoxelScene = false;

  int maxClusters = 10;
  /**
   * The maximum number of iterations to perform when looking for planar features.
   */
  int maxPlaneSegmentationIterations = 50;
  /**
   * The maximum distance that a point can be from a planar feature to be considered part of that
   * planar feature.
   */
  float segmentationDistanceThreshold = 0.01;
  /**
   * The percentage of the scene to analyze (pass onward to clustering).
   * The clustering algorithm will continue to remove planar features until this condition
   * is satisfied.
   *
   * 1.0 = the entire scene is one object
   * 0.0 = nothing will be analyzed
   */
  float percentageToAnalyze = 0.2;
  /**
   * The distance between points in the voxel grid (used to clean up the point cloud and make
   * it well-behaved for further analysis). If this value is too small, the grid will be too fine.
   * there won't be enough integers to provide indices for each point and you will get errors.
   */
  float voxelLeafSize = 0.005;
  /**
   * The maximum distance between points in a cluster (used in the Euclidean
   * clustering algorithm).
   */
  float clusterTolerance = 0.03;
  /**
   * Clusters with less than this number of points won't be analyzed. Usually this is used to
   * filter out small point clouds like anomalies, noise, bits of whatnot in the scene, etc.
   */
  int minClusterSize = 200;
  /**
   * Clusters that have more than this number of points won't be analyzed. The assumption
   * is that this is either a) too computationally intensive to analyze, or b) this is just a 
   * background object like a wall which should be ignored anyway.
   */
  int maxClusterSize = 2000;

  ///input is stored here
  PCP inputCloud;
  ///used as intermediate step for cloud processing
  PCP processCloud;

  std::vector<PCP> clusters;

  std::string originalCloudFrame = "";
  std::string cloudTopicName = "";

  /*================================================*/
  /* FILTERING STEPS (FUNCTIONS) */
  /*================================================*/
public:

  /**
   * Create a voxel grid based on point cloud data. See
   * http://www.pointclouds.org/documentation/tutorials/voxel_grid.php and
   * http://docs.pointclouds.org/1.7.1/classpcl_1_1_voxel_grid.html.
   * @param  loose    unstructured (not on a grid) point cloud
   * @param  gridSize the distance between voxels in the grid
   * @return          the points of the voxel grid created from the input
   */
  PCP& voxelGridify(PCP &loose, float gridSize);

  /**
   * Segment out planar clouds. See
   * http://pointclouds.org/documentation/tutorials/planar_segmentation.php
   * @param  input             the point cloud from which to remove planes
   * @param  maxIterations     maximum iterations for clustering algorithm.
   * @param  thresholdDistance how close a point must be to hte model in order to be considered
   *  an inlier.
   * @param  percentageGood    keep removing planes until the amount of data left is less than this
   *  percentage of the initial data.
   * @return                   the point cloud with primary planes removed as specified.
   */
  PCP& removePrimaryPlanes(PCP &input, int maxIterations, float thresholdDistance, float percentageGood);

  PCP& clipByDistance(PCP &unclipped,
      float minX, float maxX, float minY, float maxY, float minZ, float maxZ);

  /**
   * Euclidean clustering algorithm. See
   * http://www.pointclouds.org/documentation/tutorials/cluster_extraction.php
   * @param input            the cloud to cluster
   * @param clusterTolerance the maximum distance between points in a given cluster
   * @param minClusterSize   clusters of size less than this will be discarded
   * @param maxClusterSize   clusters of size greater than this will be discarded
   * @return                 a vector of point clouds, each representing a cluster from the clustering
   *  algorithm.
   */
  std::vector<PCP> cluster(PCP &input, float clusterTolerance, int minClusterSize, int maxClusterSize);

  PointCloudUtils();

  void cb_process(sensor_msgs::PointCloud2ConstPtr point_cloud);
};

#endif