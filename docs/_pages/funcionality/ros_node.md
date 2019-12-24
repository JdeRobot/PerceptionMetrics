---
layout: home
title: ROS Node
permalink: /functionality/ros_node/


sidebar:
  nav: "functionality"
---

DetectionSuite Deployer functionality is provided as a **ROS Node**.
This node uses some input images stream (it could be a video) and 
makes inferences over the images passed, detecting the objects in realtime. 
Currently it only supports tensorflow as inferencer, but a broader support is expected
in the future.

To use DetectionSuite ROS Node, it needs to subscribe to a rostopic. It would publish images
that the node would infer.

In the build/devel directory inside DetectionSuite folder, execute:

```
source setup.bash
```


While the video is playing, start the ros node with the following command:

```
rosrun detection_suite test _topic:=<topic name> _configfile:=<path/to/configfile>
```

*_topic* is the topic detection suite is suscribed to and *_confilefile* the configuration file with the inferencer, class names...
The detections are publishes at *my_topic*. Run the following command to see available topics:

```
rostopic list
```

To check the detections finally run the following command that should list the objects detected with their position in the image and the confidence
for the detection.

```
rostopic echo /my_topic
```

# Use example

{% include video id="xvPaFZD5EoY" provider="youtube" %}

In the example, the node is subscribed to a stream using [video_stream_opencv ROS driver](https://github.com/ros-drivers/video_stream_opencv).
In this case, the only needed file is video.launch, where the local video file should be set, so it detects it.
After this stream is launched using 

```
roslaunch video_stream_opencv video_file.launch
```

a window should pop up with the video.
