Publish face detections using a vision_msgs/Detection2DArray.

The face detections are calculated using the excellent face detector included in
the opencv_apps package. This example simply converts the face detection message
into a vision_msgs format.

This is not meant to suggest that opencv_apps/FaceArrayStamped is not a good
message format, rather, it's simply to test the Detection2DArray functionality
with real data and ensure that it's capable of representing a diverse set of
possible detection methods.

# Usage

You'll have to have `opencv_apps` installed. Run the face detector using the
included launch file:

```
roslaunch face_detection_bridge face_detection_bridge.launch
```

However, this probably won't do anything by itself, because you need to specify
the `image` param, which tells the face detector which topic to listen to images
on. For example, if you're using a depth camera like the ASUS Xtion Pro, you
could run:

```
roslaunch face_detection_bridge face_detection_bridge.launch image:=
```

Then, you can `rostopic echo /face_detection_bridge/detection_array` to view
the results. In the results, class 0 is a face, and class 1 is an eye.