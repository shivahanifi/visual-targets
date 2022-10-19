# Developing the Demo Code


## Table of Contents
1. [Road Map](#road-map)
2. [Bounding Box Extraction](#bounding-box-extraction)
   
    2.1. [Test Set](#test-set)

    2.2. [Using OpenPose Keypoints](#using-openpose-keypoints)



## Road Map
The goal is to add specific features to the demo code. The first step is to replace the manual bounding box extraction with an automatic one. To do so, the code structure has to change and YARP needs to be included. Additionally, as an output attention point or attention bounding box is added. The ultimate structure in mind is as shown below.

![Roadmap](Img/roadmap.jpg)

## Bounding Box Extraction
### Test Set
  
There is a [testset]() provided in order to test the initial code. This set contains 3 different folders: `board_images_human` for the 30 images recorded from the camera , `board_images_human_openpose_images` for the 30 skeleton images as an output of the `yarpOpenPose`, and `board_images_human_openpose_json` which represents the keypoints for the skeletons in a JSON format. The information in the JSON file are important to extract the bounding box. Use an [online viewer](https://jsonformatter.org/json-viewer) to understand the JSON file vividly. 

#### JSON File Explained

JSON (JavaScript Object Notation) is an open standard file format and data interchange format that uses human-readable text to store and transmit data objects consisting of attribute–value pairs and arrays. In this case a JSON file for each image represents lists that contain several lists in them. To be more clear, if you consider the file `00000000_keypoints.json` it contains a list of `people` which contains 9 lists:
- `person_id`
- `pose_keypoints_2d`
- `face_keypoints_2d`
- `hand_left_keypoints_2d`
- `hand_right_keypoints_2d` 
- `pose_keypoints_3d`, 
- `face_keypoints_3d`
- `hand_left_keypoints_3d`
- `hand_right_keypoints_3d`

With this file only `pose_keypoints_2d` and `face_keypoints_2d` contains information. For the other lists, since the related flags were not enabled they are empty.
- `pose_keypoints_2d`: There are 25 keypoints in a skeleton as shown below. For each keypoint its position (x y) and confidence level are recorded. Therefore, this list contains 75 (25*3) elements. 

    To be more clear, element 0 in the list (291.568) is the x for the nose, element 1 (171.459) is the y for the nose and element 3 (0.862217) is the confidenece level for the nose.

    <img src="Img/pose_keypoints.png" alt="pose" width="250"/>

- `face_keypoints_2d`: There are 70 keypoints in a face as shown below. As mentioned, For each keypoint its position (x y) and confidence level are recorded. Therefore, this list contains 210 (70*3) elements.

    <img src="Img/face_keypoints.png" alt="face" width="400"/>
  
### Using OpenPose Keypoints
- `read_openpose_from_json`
  
    In order to use the keypoint information from the JSON file, a function from the [leftright-gaze-estimator-demo](https://github.com/MariaLombardi/leftright-gaze-estimator-demo/blob/master/src/functions/utilities.py) will be used.
    This function loads the JSON file into a Python object. Then, for both pose and face extracts the confidenses and position.

    As a result of `append, delet` and `zip` the indices of the output exactly matches the indices of the keypoints on the image. i.e. poses[0] contains the position of keypoint 0 which is the nose, and conf_poses[0] represents the confidence level of the nose.


- `compute_centroid`
    
    This function is also defined in [leftright-gaze-estimator-demo](https://github.com/MariaLombardi/leftright-gaze-estimator-demo/blob/master/src/functions/utilities.py). It computes the mean of X and Y which will be used when creating the bounding box. 

    Note that `IMAGE_WIDTH` and `IMAGE_HEIGHT` should be defined as constants.

- `get_openpose_bbox`
    
    To extract the bounding box, a function from [face-recogniser-demo](https://github.com/MariaLombardi/face-recogniser-demo/blob/main/src/functions/utilities.py) will be used. This function adds margins to the centroid and creates the bounding box.

    Note that `JOINTS_POSE_FACE` should be defined as a constant and alòso the `joint_set(p)` function.