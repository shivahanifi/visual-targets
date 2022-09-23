## Collecting input data
### Intel RealSense Viewer
This is one way to record video/image with the RealSense Camera. To do so, install the [RealSense SDK](https://www.intelrealsense.com/sdk-2/). One of the tools is the `Viewer`. Find the user's guide for the viewer [here](https://www.intelrealsense.com/download/7144/).
After installation initiate the viewer:
```
realsense-viewer
```
For this task you can record the RGB camera. However, the output of this recording is in the `.bag` format, which can be transformed into JPEG using [rosbag](http://wiki.ros.org/rosbag/Tutorials/Exporting%20image%20and%20video%20data).

### YARP

In order to use YARP, the YARP RealSense Device should be among the YARP devices. Check it with 
```
yarpdev --list
```
In case it is not among them, add it following the link [yarp-device-realsense2](https://github.com/robotology/yarp-device-realsense2).
