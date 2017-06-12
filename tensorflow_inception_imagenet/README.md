Publishes 2D classifications based on a ROS Image topic, such as a camera
stream.

The neural network implementation uses the Inception example from the Tensorflow
models repository, with added ROS hooks.

# Usage

It's highly recommended to use a `virtualenv`, Docker, or some other
virtualization solution to avoid installing a lot of global Python packages.

If you have an nVidia GPU, you may also want to install CUDA and other GPU
dependencies as described in the
[TensorFlow installation tutorial](https://www.tensorflow.org/install/).

Install the required python packages by navigating to the
`tensorflow_inception_imagenet` root directory and running

```
pip install -r requirements.txt
```

Next, run the classifier:

```
rosrun tensorflow_inception_imagenet tensorflow_classifier.py
```
This will download the Inception network weights and
the ImageNet class labels, as well as upload the class database to the ROS
parameter server.

By default, the node will listen for images at the topic name
`/image/rgb/image_raw`. You can specify a custom `image_topic` parameter on the
command line to work with your ROS setup. You can also specify a custom topic
to publish the classifications on, using the `classification_topic` param. If
you change this topic, be sure to change it on the listener below as well.

The classifier only publishes the top 5 recognition results to ROS, but this
could easily be changed to any value you like (such as only sending the top
result). When browsing the Python source file, the ROS-specific parts of the
code are prefixed by a `# ROS:` comment so you can find them easily.


Finally, run the listener:
```
rosrun tensorflow_inception_imagenet tensorflow_listener.py
```
This will listen to the messages provided by the classifier, use the metadata
database stored on the parameter server to get the human-readable names of the
ImageNet classes, and print the results to the console.

As noted above, you can override the topic on which to listen for
classifications using the `classification_topic` parameter.

If you open `tensorflow_listener.py`, you'll see that it's under 50 lines long.
Also, the listener has no knowledge of TensorFlow or Inception, which is great:
the classifier could be running on a high-performance machine somewhere, and the
listener can listen from any other machine.