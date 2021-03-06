#!/usr/bin/env python

# ==============================================================================
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example showing how to use Tensorflow with ROS and the new vision_msgs
format.

Uses the Inception example from the Tensorflow models repository, with added
ROS hooks. This may not be the most accurate classifier for household or office
environments, especially since it is trained on an extremely diverse set of
objects, such as various animal species. However, it serves to illustrate the
usage of TensorFlow and ROS together, using vision_msgs as a bridge.

the "# ROS" comments mark important ROS integration areas.

Modification 2017, Adam Allevato

ORIGINAL DESCRIPTION

Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import os
import re
import sys
import tarfile
from copy import deepcopy

import numpy as np
from six.moves import urllib
import tensorflow as tf


from node_lookup import NodeLookup
# ROS: includes
import rospy, cv_bridge
from vision_msgs.msg import Classification2D, ObjectHypothesis, VisionInfo
from std_msgs.msg import Header
from sensor_msgs.msg import Image

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

def create_graph(model_dir):
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


class TensorflowClassifier:
  def __init__(self):

    # ROS: set up ROS and load parameters
    self.image_msg = Image()
    rospy.init_node("tensorflow_example", anonymous=False)
    image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw')
    classification_topic = rospy.get_param(
        '~classification_topic',
        '/classification')
    vision_info_topic = rospy.get_param(
        '~vision_info_topic',
        '/vision_info')
    database_param = rospy.get_param(
        '~classification_topic',
        '/tensorflow_database')

    rospy.Subscriber(image_topic, Image, self.cb_image_message)
    rospy.loginfo(
        "waiting for images to be published to {}...".format(image_topic))
    pub = rospy.Publisher(classification_topic, Classification2D, queue_size=10)
    vision_info_pub = rospy.Publisher(vision_info_topic,
                                      VisionInfo, queue_size=1)
    vision_info = VisionInfo()
    vision_info.database_location = database_param
    vision_info.method = "TensorFlow Inception detector, trained on ImageNet"
    vision_info.database_version = 0
    rate = rospy.Rate(10)
    # Now run inference

    # Creates graph from saved GraphDef.
    create_graph(FLAGS.model_dir)
    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()
    node_lookup.from_file(FLAGS.model_dir)
    node_lookup.to_rosparam(database_param)

    with tf.Session() as sess:
      image_seq = 1
      softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

      # ROS: loop until node is killed
      while not rospy.is_shutdown():
        # ROS: Wait for new image message to arrive
        if ((self.image_msg.header.seq <= image_seq) or
              (len(self.image_msg.data) <= 2)):
          rate.sleep()
          vision_info_pub.publish(vision_info)
          continue
        image_seq = self.image_msg.header.seq

        # ROS: create an empty classification message and fill metadata
        classification = Classification2D()
        classification.header = Header()
        classification.header.stamp = rospy.get_rostime()
        classification.source_img = deepcopy(self.image_msg)
        # ROS: convert image from ROS format to numpy array, and run it through
        # the Tensorflow network
        image_arr = np.array([ord(b) for b in classification.source_img.data],
                             ndmin=3)
        image_data = np.reshape(image_arr, (classification.source_img.width,
                                classification.source_img.height, 3))
        predictions = sess.run(softmax_tensor,
                               {"DecodeJpeg:0": image_data})

        # to load an image from a file instead:
        # image_data =tf.gfile.FastGFile(os.path.join(FLAGS.model_dir,
        #                                             'cropped_panda.jpg'),
        #                                'rb').read()
        # predictions = sess.run(softmax_tensor,
        #                        {'DecodeJpeg/contents:0': image_data})

        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]

        os.system("clear")
        rospy.loginfo("Tensorflow Classification results: ")
        for node_id in top_k:
          human_string = node_lookup.id_to_string(node_id)
          score = predictions[node_id]
          rospy.loginfo('    %s (score = %.5f)' % (human_string, score))

          # ROS: add results to classification message
          result = ObjectHypothesis()
          result.id = node_id
          result.score = score
          classification.results.append(result)

        # ROS: publish the classification message
        pub.publish(classification)
        rate.sleep()

  def cb_image_message(self, msg):
    self.image_msg = msg

def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()
  tfc = TensorflowClassifier()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
