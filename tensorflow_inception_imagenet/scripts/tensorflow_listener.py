#!/usr/bin/env python

"""Listens to classifcations using the new vision_msgs format, and looks up
the class definitions from the ROS parameter server to understand class names.
"""

import rospy
from vision_msgs.msg import Classification2D, VisionInfo

from node_lookup import NodeLookup

class TensorflowListener:
  def __init__(self):
    # create a lookup table from IDs to human-readable class names
    self.node_lookup = NodeLookup()

    # ROS: set up node
    rospy.init_node("tensorflow_listener", anonymous=False)
    rate = rospy.Rate(10)

    # ROS: receive vision metadata to know where to find the NodeLookup database
    vision_info_topic = rospy.get_param(
        '~vision_info_topic',
        '/vision_info')

    self.classifier_name = ""
    self.database_param = ""
    rospy.Subscriber(vision_info_topic, VisionInfo, self.cb_vision_info)
    while self.database_param == "" and not rospy.is_shutdown():
      rate.sleep()
    if rospy.is_shutdown():
      exit()

    rospy.loginfo("got vision information from classifier: " +
                  self.classifier_name + ", database information stored at " +
                  self.database_param)
    self.node_lookup.from_rosparam(self.database_param)

    # ROS: set up listener
    classification_topic = rospy.get_param(
      '~classification_topic',
      '/tensorflow_inception_imagenet/classification')

    rospy.Subscriber(classification_topic, Classification2D,
                     self.cb_classification)

    rospy.loginfo(
        "waiting for classifications to be published to {}...".format(
            classification_topic))
    while not rospy.is_shutdown():
      rate.sleep()

  # ROS: the callback for when a classification message is received
  def cb_classification(self, msg):
    rospy.loginfo("classification received:")
    for result in msg.results:
      rospy.loginfo("    {}: {}".format(
          self.node_lookup.id_to_string(result.id),
          result.score))

  # ROS: the callback for when vision pipeline information is received
  def cb_vision_info(self, msg):
    rospy.loginfo("vision info received")
    self.classifier_name = msg.method
    # note: it's assumed here the the database location is a ROS parameter
    # server, but this is not necessary. It could be a string representing
    # an XML/JSON/YAML/etc. structure, a file path, a database URL, etc.
    # the implementation is left up to the individual classifier.
    self.database_param = msg.database_location

if __name__ == '__main__':
  tfc = TensorflowListener()