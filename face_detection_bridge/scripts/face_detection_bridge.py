#!/usr/bin/env python

"""Translate from the opencv_apps face detector to the new vision_msgs format.

This is probably not the best use of the vision_msgs format, but it does show
off how a multi-object detection message (Detection2DArray) can be used. The
FaceArrayStamped is probably a good message to continue to use for the specific
problem of face detection, if that is your final use case."""

import rospy
from vision_msgs.msg import CategoryDistribution, Detection2D, Detection2DArray
from opencv_apps.msg import Rect, Face, FaceArrayStamped
from std_msgs.msg import Header

class FaceDetectionBridge:
    def cb_faces(self, face_array):
        if self.pub is None:
            return
        # This entire next section is just converting from FaceArrayStamped to
        # Detection2DArray for eyes and faces.
        # This section could also fill the optional detection.source_img fields
        # with cropped images of the faces and eyes, but this isn't necessary
        # for the demo's purpose.
        detection_arr = Detection2DArray()
        for face in face_array.faces:
            detection = Detection2D()
            detection.header = Header()
            detection.bbox_size_x = face.face.width
            detection.bbox_size_y = face.face.height
            detection.pose.x = face.face.x
            detection.pose.y = face.face.y

            # create a single-bin histogram for the category distribution:
            # it's a face, 100% sure, in all cases
            cat = CategoryDistribution()
            cat.ids.append(0) # face
            cat.scores.append(1.0)
            detection.results = cat
            detection_arr.detections.append(detection)

            for eye in face.eyes:
                detection = Detection2D()
                detection.header = Header()
                detection.bbox_size_x = eye.width
                detection.bbox_size_y = eye.height
                detection.pose.x = eye.x
                detection.pose.y = eye.y

                # create a single-bin histogram for the category distribution:
                # it's a eye, 100% sure, in all cases
                cat = CategoryDistribution()
                cat.ids.append(1) # eye
                cat.scores.append(1.0)
                detection.results = cat
                detection_arr.detections.append(detection)
        self.pub.publish(detection_arr)

    def __init__(self):
        rospy.init_node('face_detection_bridge', anonymous=True)

        self.pub = rospy.Publisher('~detection_array', Detection2DArray, queue_size=10)
        sub = rospy.Subscriber("/face_detection/faces", FaceArrayStamped, self.cb_faces)

        rospy.loginfo("Started face detection bridge.")
        rospy.spin()

if __name__ == '__main__':
    fdb = FaceDetectionBridge()