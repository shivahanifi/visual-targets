import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import shutil
import xml.etree.ElementTree as ET

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

root_path ='/Users/shiva.hnf/Desktop/ObjectAttention_augmentation'

density_list = []
total_frame_count = 0

# Iterate over participants
for participant_folder in os.listdir(root_path):
    if '.DS' not in participant_folder:
        participant_path = os.path.join(root_path, participant_folder)
    
        # Check if the item in the root folder is a directory
        if os.path.isdir(participant_path):
            #print(f"participant_path : {participant_path}")         
            # Iterate over sessions
            for session_folder in os.listdir(participant_path):
                if '.DS' not in session_folder:
                    session_path = os.path.join(participant_path, session_folder)
                
                    # Check if the item in the participant folder is a directory
                    if os.path.isdir(session_path):
                        #print(f"session_path : {session_path}") 
                        # Iterate over settings
                        for setting_folder in os.listdir(session_path):
                            if '.DS' not in setting_folder:
                                setting_path = os.path.join(session_path, setting_folder)
                                # Number of objects in the previous object folder
                                frame_count = 0
                                # Check if the item in the session folder is a directory
                                if os.path.isdir(setting_path):
                                    #print(f"setting_path : {setting_path}")
                                    # Iterate over target objects
                                    for object_folder in os.listdir(setting_path):
                                        # Path to XML annotation
                                            xml_file = glob.glob(os.path.join(setting_path, '*.xml'))
                                            xml_path = os.path.join(setting_path, xml_file[0])
                                            if '.DS' not in object_folder and 'xml' not in object_folder:
                                                object_path = os.path.join(setting_path, object_folder)
                                                
                                                # Check if the item in the setting folder is  a directory
                                                if os.path.isdir(object_path):
                                                    canvas = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
                                                    #print(f"object_path : {object_path}")
                                                    for obj_subfolder in os.listdir(object_path):
                                                        if '.DS' not in obj_subfolder:
                                                            obj_subfolder_path = os.path.join(object_path, obj_subfolder)
                                                            if os.path.isdir(obj_subfolder_path):    
                                                                #print(f"obj_subfolder_path : {obj_subfolder_path}")
                                                                if 'rgb' in obj_subfolder:
                                                                    obj_frames = [frame for frame in os.listdir(obj_subfolder_path) if frame.lower().endswith('.ppm')]
                                                                    obj_frames.sort()
                                                                    num_frames = len(obj_frames)
                                                                    tree = ET.parse(xml_path)
                                                                    root = tree.getroot()
                                                                    objects = root.findall('object')
                                                                    for obj in objects:
                                                                        if obj.find('name').text == str(object_folder) :
                                                                            xmin_obj = int(obj.find('bndbox/xmin').text)
                                                                            xmax_obj = int(obj.find('bndbox/xmax').text)
                                                                            ymin_obj = int(obj.find('bndbox/ymin').text)
                                                                            ymax_obj = int(obj.find('bndbox/ymax').text)
                                                                total_frame_count = total_frame_count + num_frames       
                                                                canvas[ymin_obj:ymax_obj, xmin_obj:xmax_obj] += 1
                                                                total_canvas_object = canvas * num_frames
                                                                density_list.append(total_canvas_object)
                                                                print('canvases for the participant {}, setting {}, object {} created'.format(participant_folder, setting_folder, object_folder))


# Sum the grayscale images
sum_image = np.zeros_like(density_list[0], dtype=np.float32)
for image in density_list:
    sum_image += image

object_location_density = (sum_image/total_frame_count * 255).astype(np.uint8)

# Apply thresholding to make white regions more specific
_, thresholded_image = cv2.threshold(object_location_density, 0, 255, cv2.THRESH_BINARY)

# Apply adaptive thresholding to enhance white regions
#_, thresholded_image = cv2.threshold(object_location_density, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


cv2.imshow('object location density',thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()                                                             
                                                                
                                                                
                                    
                                                                    

                                                                    
                                                                    
                                                                