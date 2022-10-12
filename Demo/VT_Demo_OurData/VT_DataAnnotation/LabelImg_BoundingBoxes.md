# Annotating Images
In this step, the aim is to annotate the collected input images. [LabelImg](https://github.com/heartexlabs/labelImg) tool will be used. 
## Installing LabelImg
You can easily install and run LabelImg with the command:
```
pip3 install labelImg
labelImg
```
## Annotating the Data
The input images have been selected from the raw input data. There are 4 input sets. For the set that contains two people, the annotation have been done 2 times, once for each person.
The aim here is to annotate the head bounding boxes for each image. To do so, open the application and through the left menu bar, `Open Dir`, open the directory your images are in. After each annotation you have to save the result. Annotating each image will result in a XML file for which you have to specify the diroctory to be saved (Use `Change Save Dir`). 

To speed up the process use the shortcuts mentioned in the LabelImg's repository.

## Converting to TXT
Output of the LabelImg is a XML file per each image. However, in the demo we need a TXT file that contains the image name, and information related to the bounding box (xmin,ymin,xmax,ymax). The `xmltotxt.py` converts XML files into a single TXT file in a way we desire. The result of this application is collected in the `bndBox_TXT` folder.