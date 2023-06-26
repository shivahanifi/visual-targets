# Finetuning the VTD architecture with ObjectAttention dataset
Here I will document the steps to finetune the VTD architecture using the [ObjectAttention](https://drive.google.com/drive/folders/1qr71qBfq6rkG4SXOFjZvb3GhOxx-3NRv?usp=sharing) dataset.

## Table of contents
- [Finetuning the VTD architecture with ObjectAttention dataset](#finetuning-the-vtd-architecture-with-objectattention-dataset)
  - [Table of contents](#table-of-contents)
  - [Feasibility of retraining](#feasibility-of-retraining)
  - [Finetuning VTD using the VideoAttentionTarget dataset](#finetuning-vtd-using-the-videoattentiontarget-dataset)
    - [Errors](#errors)
  - [Preparing ObjectAttention dataset to finetune VTD](#preparing-objectattention-dataset-to-finetune-vtd)
  - [Trainining on ObjectAttention](#trainining-on-objectattention)
    - [Initial weights for retraining](#initial-weights-for-retraining)
    - [**Training evaluation for 3 epochs, batch size 16 and chunk size 3**](#training-evaluation-for-3-epochs-batch-size-16-and-chunk-size-3)
    - [**Training evaluation for 15 epochs, batch size [4,8,16] and chunk size [2,3,4]**](#training-evaluation-for-15-epochs-batch-size-4816-and-chunk-size-234)
  - [Training on ObjectAttention augmented with VideoAttentionTarget](#training-on-objectattention-augmented-with-videoattentiontarget)


## Feasibility of retraining
The first step is to analyze the feasibility of retraining the VTD architecture. Initial training of VTD is performed in a two-step process. First, the model is globally trained on the GazeFollow dataset until convergence. Second, it is subsequently trained on the VideoAttentionTarget dataset, while freezing the layers up
to the Encode module to prevent overfitting. The code for both steps of the training is available:
- [train_on_gazefollow.py](https://github.com/ejcgt/attention-target-detection/blob/master/train_on_gazefollow.py)
- [train_on_videoatttarget.py](https://github.com/ejcgt/attention-target-detection/blob/master/train_on_videoatttarget.py)

The hyper parameters for each of the training sessions are specified as:
<img src=img/hyperparam.png>

Moreover, For each dataset they have defined a custom PyTorch dataset class in [dataset.py](https://github.com/ejcgt/attention-target-detection/blob/master/dataset.py) where they preprocess the data and apply several data augmentation techniques to the input images. The code also generates the ground truth gaze heatmap based on the gaze coordinates.

Since all the required information for retraining are available, it is feasible to finetune the VTD to improve the performance.

## Finetuning VTD using the VideoAttentionTarget dataset 
Before finetuning the model with our dataset I try to regenerate what the authors did with the [VideoAttentionTarget dataset](https://www.dropbox.com/s/8ep3y1hd74wdjy5/videoattentiontarget.zip?dl=0). The steps and related errors are listed below:
1. Downloaded the [VideoAttentionTarget dataset](https://www.dropbox.com/s/8ep3y1hd74wdjy5/videoattentiontarget.zip?dl=0). Note that the zip file needs roughly 16GB of space.
   - Data structure: the folder contains `images` and `annotations`. In the `images`, there are folders that specify a TV show, and inside each of the folders there are clips folders which contains related frames. The annotation on the other hand, is seperated to the `train`, abd `test` folders, abd for each of them follows the structure of the `images` folder.
2. Since the VTD and all its developements are being run in the docker environment  on the IITICUBLAP156 laptop, for the docker to see the dataset, a folder is shared between a docker and local host. To do so, the docker was being run using the code below:
   ```
   nvidia-docker run -v /home/icub/Users/emaiettini/shiva_vtd/vtd_retrain:/projects/attention-target-detection --rm -it --privileged --gpus all  --privileged -v /dev:/dev -e QT_X11_NO_MITSHM=1 -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --hostname dockerpc --network=host --pid=host visual_target_attention bash
   ``` 
3. The path for the train and test data are specified in the `config.py` file. To that end, the variables of `videoattentiontarget_train_data`, and `videoattentiontarget_train_label` are given the related path. Note that the path from the docker should be provided otherwise since the code is being run from inside the docker would not be able to find the data.
4. Finally, you can run the [train_on_videoatttarget.py](https://github.com/ejcgt/attention-target-detection/blob/master/train_on_videoatttarget.py), to start the training.
   
### Errors
- Memory:
  
    The VideoAttentionTarget dataset is a large dataset and with the data augmentation applied the training requires noticable memory space. The 31 GB memory of the alienware laptop was not enough and the training process was getting aborted at some point. To deal with this issue I tried to reduce the memory load in few steps:
  1. Reducing the batch size from 16 to 1. This step alone did not solve the issue.
  2. Specifying a shared memory size of 31G for the docker `--shm-size=31g` did not solve the problem either.
  3. Reducing the size of the dataset for training. Removing 25 folders and leaving 15 (only one of the folders "band of brothers" contains 1.871 frames, therefore roughly around 22500 frames used for training in this case). This step helped to pass the memory issue and start the training process.
  4. Eventhough the training process was completed and the logs recorded, there is still a `THCudaCheck error` remaining. 

## Preparing ObjectAttention dataset to finetune VTD
To use the [ObjectAttention](https://drive.google.com/drive/folders/1qr71qBfq6rkG4SXOFjZvb3GhOxx-3NRv?usp=sharing) for retraining VTD there are some preprocessing steps required.

- Data annotaion
  
  The training requires the annotation of the head bounding box for each frame as well as the pixel position of the gaze target as `gaze_x`, `gaze_y`. To that end, the ObjectAttention dataset is annotated using the [labelImg](https://github.com/heartexlabs/labelImg) tool. Note that annotating one frame from each of the folders is enough since the participant is standing almost still for each of the target objects.

- Data structure

    The goal is to structure the ObjectAttention dataset as the VideoAttentionTarget dataset. To that end, I structured it in sessions which are considered as shows, and the settings as clips. Therefore, using [change_frame_name.py](), I renamed all the frames of each of the settings to be in sequence. Subsequently, using [VTD_retrain_annotationTXT.py](), I generated a single TXT file for each setting, containing the annotations in the form of:
    ```
    [frame x_min y_min x_max y_max gaze_x gaze_y]
    ```

    The restructured dataset can be found in [ObjectAttention_retrain](https://drive.google.com/file/d/1ZeH0mFlE2gVtTLYsw48ERorzn0QQC1rS/view?usp=sharing).

## Trainining on ObjectAttention
To train on the ObjectAttention dataset the code scripts relatd to the data preprocessing and training must be modified. 

- Note that you first need to modify the `config.py` file, and provide the path for the related images and labels. Since the code is being run from inside the docker, even though a shared folder is being used, you should provide the path from inside the docker for the images to be accessible.
- a new class of dataset is defined for the ObjectAttention dataset in the [dataset.py](https://github.com/shivahanifi/visual-targets/blob/main/VTD_retrain/src/dataset.py). For the images to be readable, the `__getitem__` method is modified, such that when reading the TXT annotation file, since the TXT annotation files I am providing has the header, the `header option is set to 0`. Moreover, the name of the images should be taken completely and as a string. Therefore, the type is specified as string. 
  ```
  dtype={'frame':str}
  ```
  Also `.ppm` extension is added to the image names when reading them.
  
- Moreover, for the main training code [train_on_objectattention.py](https://github.com/shivahanifi/visual-targets/blob/main/VTD_retrain/src/train_on_objectattention.py), the new class related to the ObjectAttention dataset is included and 

### Initial weights for retraining
1. In order to retrain the architecture, I first used the [initial_weights_for_temporal_training.pt](https://www.dropbox.com/s/s9y65ajzjz4thve/initial_weights_for_temporal_training.pt) as initial weights. Which are initial weights provided by the VTD authors. Evaluating the training using the [eval_on_objectattention.py](https://github.com/shivahanifi/visual-targets/blob/main/VTD_retrain/src/eval_on_objectattention.py) resulted in AUC score of 0.7803 and distance of 0.1078. 
2. The second step is to finetune the model using the [model_videoatttarget.pt](https://www.dropbox.com/s/ywd16kcv06vn93x/model_videoatttarget.pt) as initial weights.

At this step to have the original batch size of 16, an external GPU is required.

### **Training evaluation for 3 epochs, batch size 16 and chunk size 3**
To evaluate the training I use [eval_on_objectattention.py]() and record the AUC and dist parameters.

- AUC
  
  “Area Under the Curve” (AUC) of the “Receiver Operating Characteristic” (ROC). The Receiver Operator Characteristic (ROC) curve is an evaluation metric for binary classification problems. The Area Under the Curve (AUC) is the measure of the ability of a binary classifier to distinguish between classes and is used as a summary of the ROC curve.

  The higher the AUC, the better the model’s performance at distinguishing between the positive and negative classes.

- dist

  The L2 distance between the groundtruth and the argmax point.

  |   Initial weights for training	|  Training dataset  	| 	Weights used for evaluation| AUC |  dist |	
  |---	|---	|---	|---	|---|
  |initial_weights_for_temporal_training.pt|  VideoAttentionTarget 	|  model_videoatttarget.pt  |0.8915 	|  0.1682| 
  |initial_weights_for_temporal_training.pt|  ObjectAttention 	|  epoch_3_weights.pt |0.8569 	|  0.0821|
  |initial_weights_for_temporal_training.pt|  ObjectAttention 	|  epoch_3_weights.pt |0.8334 	|  0.0876|
  |initial_weights_for_temporal_training.pt|  ObjectAttention 	|  epoch_3_weights.pt |0.8017 	|  0.0862|
  |  model_videoatttarget.pt 	|  ObjectAttention |epoch_3_weights.pt	|  0.8686 	|  0.1090 	|
  |  model_videoatttarget.pt 	|  ObjectAttention |epoch_3_weights.pt	|  0.8520 	|  0.1008 	|
  |  model_videoatttarget.pt 	|  ObjectAttention |epoch_3_weights.pt	|  0.8207 	|  0.0919 	|

### **Training evaluation for 15 epochs, batch size [4,8,16] and chunk size [2,3,4]**

To have a better understanding of the training I evaluated the training with 15 epochs and different values for the batch size and chunk size.

Note that I modified the evaluation code, such that it iterates over all the epoch weights and reports AUC and dist. This code can be found in [eval_on_objectattention_multiepoch.py](https://github.com/shivahanifi/visual-targets/blob/main/VTD_retrain/src/eval_on_objectattention_multiepoch.py).
To automatize the process, I used a bash script [run_experiments.sh](https://github.com/shivahanifi/visual-targets/blob/main/VTD_retrain/src/run_experiments.sh) that runs the required commands consequently.

The visualization of the AUC and dist for all these scenarios is as follow:
 |batch size \ chunk size | 2 | 3 | 4 |	
  |---	|---	|---	|---	|
  | 4 |  <img src=img/auc_dist_bs4_cs2.png> 	| <img src=img/auc_dist_bs4_cs3.png>    | <img src=img/auc_dist_bs4_cs4.png> | 
  | 8 |  CUDA error!	|  CUDA error!  | CUDA error! |
  | 16 |  <img src=img/auc_dist_bs16_cs2.png> 	| <img src=img/auc_dist_bs16_cs3.png>| CUDA error! |


## Training on ObjectAttention augmented with VideoAttentionTarget

Adding the training images and labels of the 12 shows from VideoAttentionTarget: Veep, Three Idiots, The View, Tartuffle, Suits, Star Wars, Sherlock, Sienfield, Secret, Hearing, Friends, Coveted.
Also 5 shows from VideoAttentionTarget are added to the test sets: Jamie Oliver, MLB Interview, Survivor, Titanic, West World. This trial failed due to the differnces in image types. (ObjectAttention has .ppm images while VideoAttentionTarget contains .jpg images)