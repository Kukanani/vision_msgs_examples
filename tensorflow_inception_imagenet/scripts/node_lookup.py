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

"""Performs lookup of metadata (ImageNet class name) from numeric IDs. Used
by both the listener and classifier as a basic "metadata database". """


import os
import os.path
import re
import xmltodict
from dicttoxml import dicttoxml
import tensorflow as tf
import rospy

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    self.node_lookup = {}


  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

  def from_file(self, model_dir):
    """Loads a human readable English name for each softmax node.

    """
    label_lookup_path = os.path.join(
        model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    uid_lookup_path = os.path.join(
        model_dir, 'imagenet_synset_to_human_label_map.txt')

    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    self.node_lookup = node_id_to_name

  def to_rosparam(self, database_param):
    lookup_dict = [{"id": key, "name": value}
                  for (key, value)in self.node_lookup.iteritems()]
    xml = dicttoxml(lookup_dict, custom_root="classes", attr_type=False)

    # ROS: fill the parameter server
    rospy.set_param(database_param, xml)

  def from_rosparam(self, database_param):
    # ROS: load from the parameter server
    xml = rospy.get_param(database_param)
    class_dict_list = xmltodict.parse(xml)["classes"]["item"]
    self.node_lookup = {class_dict["id"]: class_dict["name"]
                       for class_dict in class_dict_list}