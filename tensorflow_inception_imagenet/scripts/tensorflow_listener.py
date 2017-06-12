#!/usr/bin/env python

"""Listens to classifcations using the new vision_msgs format, and looks up
the class definitions from the ROS parameter server to understand class names.
"""

import rospy
from vision_msgs.msg import Classification2D

from tensorflow_classifier import NodeLookup

class TensorflowListener:
  def __init__(self):
    # create a lookup table from IDs to human-readable class names
    self.node_lookup = NodeLookup()
    self.node_lookup.from_rosparam()

    rospy.init_node("tensorflow_listener", anonymous=False)
    classification_topic = rospy.get_param('~classification_topic', '/tensorflow_inception_imagenet/classification')

    rospy.Subscriber(classification_topic, Classification2D, self.cb_classification)
    rospy.loginfo("waiting for classifications to be published to {}...".format(classification_topic))
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
      rate.sleep()

  def cb_classification(self, msg):
    rospy.loginfo("classification received:")
    for result in msg.results:
      rospy.loginfo("    {}: {}".format(self.node_lookup.id_to_string(result.id), result.score))

if __name__ == '__main__':
  tfc = TensorflowListener()