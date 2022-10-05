# Collecting Input Data

The goal here is to collect 20 input images using the RealSense camera. To properly use RealSense we can either use YARP or RealSense Viewer.

## YARP
In order to use YARP, the YARP RealSense Device should be among the YARP devices. Check it with 
```
yarpdev --list
```
In case it is not present, add it following the link [yarp-device-realsense2](https://github.com/robotology/yarp-device-realsense2).

In our case, to have all we need, a docker is used. This is a docker from the [mutual-gaze-classifier-demo](https://github.com/MariaLombardi/mutual-gaze-classifier-demo/tree/main/app) repository, which contains YARP, RealSense yarp device, OpenPose, yarpopenpose and etc.

- How to use the docker?
  
  1. Clone the repository to your local machine and build the docker. You need to build it once.
  ```
    docker build --build-arg "START_IMG=pytorch/pytorch:1.4-cuda10.1-cudnn7-devel" --build-arg "release=master" --build-arg "sbtag=Unstable" -t mutual_gaze .
  ```

  NOTE: If there is an error while building, it is probably related to the version mismatches. Check the public key and swig versions and update them in the docker file.

  2. Connect the RealSense camera before initiating the docker environment.
  3. In the terminal, head to the location in which you have stored the `demo_docker` file.
   ```
   xhost +

   nvidia-docker run --rm -it --privileged --gpus 1 -e QT_X11_NO_MITSHM=1 -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics --hostname dockerpc --network=host --pid=host mutual_gaze
   ```

   The xhost command adds or deletes host names on the list of machines from which the X Server accepts connections.

   After entring the environment you will see the  `root@dockeropc` in the terminal.
   In order to run the YARP you have to heat to the `projects` folder. If you list (ls) the folders in the projects you will see all the tools needed are included here.
   ```
   yarpserver --write
   ```
   4. To open another terminal in the docker environment, you can simply run the code below. To find the container ID, open another terminal and use the code `docker ps` to see the list of the environments and their ID.
   ```
   docker exec -it <id_container> bash
   ```
   5. To get out of the docker environment, first `Ctrl+c` and then type `exit`.

Having the yarp RealSense device installed properly, now we need an application XML file to properly connet the ports and modules. The structure we need is as below:
![structure](Img/structure.jpeg)

- `yarpdev` is the wrapper for the RealSense camera. It automatically holds open a YARP port which is called `/depthCamera/rgbImage:o` in the application file.
- [yarpdatadumper](https://www.yarp.it//v3.5/yarpdatadumper.html#yarpdatadumper_intro) acquires and stores Bottles or Images and Videos from a YARP port. It automatically opens a port that we call it `VTDataCollection:i`. This port is also used to saved the data in the folder.
- `yarpview`is used to display the input recieved from the camera. The port is called `/view/rgb`.

Find the related application in [VT_Input_Data.xml]().

NOTE: Here the modules used automatically have ports and there is no need to define ports in a seperate python file.


## Intel RealSense Viewer
This is another way to record video/image with the RealSense Camera. To do so, install the [RealSense SDK](https://www.intelrealsense.com/sdk-2/). One of the tools is the `Viewer`. Find the user's guide for the viewer [here](https://www.intelrealsense.com/download/7144/).
After installation initiate the viewer:
```
realsense-viewer
```
For this task you can record the RGB camera. However, the output of this recording is in the `.bag` format, which can be transformed into JPEG using [rosbag](http://wiki.ros.org/rosbag/Tutorials/Exporting%20image%20and%20video%20data).





