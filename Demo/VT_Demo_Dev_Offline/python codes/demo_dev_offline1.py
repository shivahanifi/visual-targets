import argparse, os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import math
from PIL import Image
from scipy.misc import imresize
from model import ModelSpatial
from utils import imutils, evaluation
from config import *

#import logging
#logging.basicConfig(filename='my.log', level=logging.DEBUG)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
JOINTS_POSE_FACE = [0, 15, 16, 17, 18]
TXT_FILE_PATH = '/Users/shiva.hnf/Documents/IIT/visual-targets/Demo/VT_Demo_Development/pose.txt'
JSON_FILES_PATH = '/Users/shiva.hnf/Documents/IIT/visual-targets/Demo/VT_Demo_Development/testset/board_images_human_openpose_json'

#Keypoint extraction
def read_openpose_from_json(json_filename):

    with open(json_filename) as data_file:
        loaded = json.load(data_file)

        poses = []
        conf_poses = []
        faces = []
        conf_faces = []

        for arr in loaded["people"]:
            conf_poses.append(arr["pose_keypoints_2d"][2::3]) #seq[start:end:step]
            arr_poses = np.delete(arr["pose_keypoints_2d"], slice(2, None, 3))
            poses.append(list(zip(arr_poses[::2], arr_poses[1::2]))) #respectively X and Y

            conf_faces.append(arr["face_keypoints_2d"][2::3])
            arr_faces = np.delete(arr["face_keypoints_2d"], slice(2, None, 3))  # remove confidence values from the array
            faces.append(list(zip(arr_faces[::2], arr_faces[1::2])))

    return poses, conf_poses, faces, conf_faces


def compute_centroid(points):
    mean_x = np.mean([p[0] for p in points])
    mean_y = np.mean([p[1] for p in points])

    if mean_x >= IMAGE_WIDTH:
        mean_x = IMAGE_WIDTH-1
    if mean_x < 0:
        mean_x = 0
    if mean_y >= IMAGE_HEIGHT:
        mean_y = IMAGE_HEIGHT-1
    if mean_y < 0:
        mean_y = 0

    return [mean_x, mean_y]


def joint_set(p):
    return p[0] != 0.0 or p[1] != 0.0


def get_openpose_bbox(pose):

    n_joints_set = [pose[joint] for joint in JOINTS_POSE_FACE if joint_set(pose[joint])]
    if n_joints_set:
        centroid = compute_centroid(n_joints_set)

        min_x = min([joint[0] for joint in n_joints_set])
        max_x = max([joint[0] for joint in n_joints_set])
        min_x -= (max_x - min_x) * 0.2
        max_x += (max_x - min_x) * 0.2

        width = max_x - min_x

        min_y = centroid[1] - (width/3)*2
        max_y = centroid[1] + (width/3)*2

        min_x = math.floor(max(0, min(min_x, IMAGE_WIDTH)))
        max_x = math.floor(max(0, min(max_x, IMAGE_WIDTH)))
        min_y = math.floor(max(0, min(min_y, IMAGE_HEIGHT)))
        max_y = math.floor(max(0, min(max_y, IMAGE_HEIGHT)))

        return min_x, min_y, max_x, max_y
    else:
        #print("Joint set empty!")
        return None, None, None, None


#collecting all the openpose json files (CHANGE THE PATH)
listOfFiles = os.listdir(JSON_FILES_PATH)


#detecting bounding boxes
f = open(TXT_FILE_PATH,'w') #Creates a new file
f.close()
for j in listOfFiles:
    poses, conf_poses, faces, conf_faces = read_openpose_from_json(j)
    min_x, min_y, max_x, max_y = get_openpose_bbox(poses)
    line_to_write = j + ',' + min_x + ',' + min_y + ',' + max_x + ',' + max_y + '\n'
    #writing to the txt file
    with open(TXT_FILE_PATH, 'a') as f:
      f.write(line_to_write)     



parser = argparse.ArgumentParser()
parser.add_argument('--model_weights', type=str, help='model weights', default='model_demo.pt')
parser.add_argument('--image_dir', type=str, help='images', default='data/demo/frames')
parser.add_argument('--head', type=str, help='head bounding boxes', default=TXT_FILE_PATH)
parser.add_argument('--vis_mode', type=str, help='heatmap or arrow', default='heatmap')
parser.add_argument('--out_threshold', type=int, help='out-of-frame target dicision threshold', default=100)
args = parser.parse_args()

def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


def run():
    column_names = ['frame', 'left', 'top', 'right', 'bottom']
    df = pd.read_csv(args.head, names=column_names, index_col=0)
    df['left'] -= (df['right']-df['left'])*0.1
    df['right'] += (df['right']-df['left'])*0.1
    df['top'] -= (df['bottom']-df['top'])*0.1
    df['bottom'] += (df['bottom']-df['top'])*0.1


    logging.debug(df)
    # set up data transformation
    test_transforms = _get_transform()

    model = ModelSpatial()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_weights)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()
    model.train(False)

    with torch.no_grad():
        for i in df.index:
            frame_raw = Image.open(os.path.join(args.image_dir, i))
            frame_raw = frame_raw.convert('RGB')
            width, height = frame_raw.size

            logging.debug(width)
            logging.debug(height)

            head_box = [df.loc[i,'left'], df.loc[i,'top'], df.loc[i,'right'], df.loc[i,'bottom']]

            head = frame_raw.crop((head_box)) # head crop

            head = test_transforms(head) # transform inputs
            frame = test_transforms(frame_raw)
            head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                                        resolution=input_resolution).unsqueeze(0)

            head = head.unsqueeze(0).cuda()
            frame = frame.unsqueeze(0).cuda()
            head_channel = head_channel.unsqueeze(0).cuda()

            # forward pass
            raw_hm, _, inout = model(frame, head_channel, head)

            logging.debug(raw_hm)
            logging.debug(inout)

            # heatmap modulation
            raw_hm = raw_hm.cpu().detach().numpy() * 255
            raw_hm = raw_hm.squeeze()
            inout = inout.cpu().detach().numpy()
            inout = 1 / (1 + np.exp(-inout))
            inout = (1 - inout) * 255
            norm_map = imresize(raw_hm, (height, width)) - inout

            #logging.debug(norm_map)

            # vis
            plt.close()
            fig = plt.figure()
            fig.canvas.manager.window.move(0,0)
            plt.axis('off')
            plt.imshow(frame_raw)

            ax = plt.gca()
            rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2]-head_box[0], head_box[3]-head_box[1], linewidth=2, edgecolor=(0,1,0), facecolor='none')
            ax.add_patch(rect)

            if args.vis_mode == 'arrow':
                if inout < args.out_threshold: # in-frame gaze
                    pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                    norm_p = [pred_x/output_resolution, pred_y/output_resolution]
                    circ = patches.Circle((norm_p[0]*width, norm_p[1]*height), height/50.0, facecolor=(0,1,0), edgecolor='none')
                    ax.add_patch(circ)
                    plt.plot((norm_p[0]*width,(head_box[0]+head_box[2])/2), (norm_p[1]*height,(head_box[1]+head_box[3])/2), '-', color=(0,1,0,1))

                    #logging.debug(args.vis_mode)
                    #logging.debug(inout)
                    #logging.debug(args.out_threshold)
                    #logging.debug(norm_p)

            else:
                plt.imshow(norm_map, cmap = 'jet', alpha=0.2, vmin=0, vmax=255)

            plt.show(block=False)
            plt.pause(1)
            plt.savefig('/home/r1-user/code_sh/new_new/attention-target-detection/data/demo/Result5/fig{0}.png'.format(i))

        print('DONE!')


if __name__ == "__main__":
    run()
