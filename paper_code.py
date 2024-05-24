import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon, intersection, union
from skimage.measure import find_contours
import os
import random
from skimage.draw import polygon
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import os


annotation_file = "../tattoo_annotations.txt"
annotations = []
match_pattern =r'(.+\.jpg) (\[.+\])'
split_pattern = r'[;\s\t\n\]]'
result = {}
with open(annotation_file) as file:
    for input_str in file.readlines():
        match = re.match(match_pattern, input_str)
        if match:
            filename = match.group(1)
            coordinates_str = match.group(2)
            coordinates_list = re.split(split_pattern, coordinates_str)
            coordinates_list = [x.strip(r"\[\]") for x in coordinates_list if x != ""]
            result[filename] = coordinates_list
        else:
            print("No matches!")
results_evaluated = {}
for k,v in result.items():
    results_evaluated[k] = [eval(x) for x in v]
dirlist = os.listdir("../tattoo_images")
keylist = results_evaluated.keys()


def polygon_to_mask(image_shape, polygon_points):
    '''

    :param image_shape: Shape of image
    :param polygon_points: Points that are inside the polygon
    :return: Binary mask of the polygon
    '''
    polygon_points = np.array(polygon_points)
    rr, cc = polygon(polygon_points[:, 1], polygon_points[:, 0], image_shape)
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[rr, cc] = 1
    return mask, rr, cc

def convert_rgb_to_binary_mask(rgb_mask):
    """
    Convert an RGB mask to a binary mask.

    Parameters:
    rgb_mask (np.array): RGB mask of shape (height, width, 3)

    Returns:
    np.array: Binary mask of shape (height, width)
    """
    # Convert RGB mask to grayscale
    gray_mask = rgb2gray(rgb_mask)
    # Threshold the grayscale image to get a binary mask
    binary_mask = img_as_ubyte(gray_mask > 0)

    return binary_mask



def random_points(mask, no_true = 1, no_false = 0):
    '''
    Selects n random points inside the mask, and m points outside in the background
    :param mask: Binary mask in the shape of the image
    :param no_true: n
    :param no_false: m
    :return: numpy array of points
    '''
    tattoo_indices = np.argwhere(mask == 255)
    background_indices = np.argwhere(mask == 0)
    tattoo_points = tattoo_indices[np.random.choice(len(tattoo_indices), no_true, replace=False)]
    background_points = background_indices[np.random.choice(len(background_indices), no_false, replace=False)]
    tattoo_points = tattoo_points[:, [1,0]]
    background_points = background_points[:, [1,0]]
    return tattoo_points, background_points


def iou(pred, ground_truth):
    pred = pred.astype(bool)
    ground_truth = ground_truth.astype(bool)
    pred = pred.flatten()
    ground_truth = ground_truth.flatten()
    intersection = np.sum(pred*ground_truth)
    union = np.sum(pred) + np.sum(ground_truth)
    iou = intersection / union if union != 0 else 0
    return iou

def fpr_fnr(pred, ground_truth):
    pred = pred.astype(int)
    ground_truth = ground_truth.astype(int)
    pred = pred.flatten()
    ground_truth = ground_truth.flatten()
    tp = np.sum(pred * ground_truth)
    fp = np.sum(pred * (1-ground_truth))
    fn = np.sum((1 - pred) * ground_truth)
    tn = np.sum((1 - pred) * (1 - ground_truth))
    fpr = fp/(fp+tn) if fp+tn != 0 else 0
    fnr = fn / (fn+tp) if fn+tp != 0 else 0
    return fpr, fnr

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "../sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

def all_combine_save_number(true=1, false=1, bounding_box = False):
    io_list = []
    fpr_list = []
    fnr_list = []
    pictures = {}
    for key, value in results_evaluated.items():
        image = cv2.imread(f"../tattoo_images/{key}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask, rr, cc = polygon_to_mask(image.shape, value)
        mask = convert_rgb_to_binary_mask(mask)
        predictor.set_image(image)
        # Create Prompt Points
        if not bounding_box:
            tattoo, background = random_points(mask, true, false)
            input_points = np.concatenate((tattoo, background))
            input_label = np.concatenate((np.ones(true), np.zeros(false)))
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_label,
                multimask_output=True,
            )
            max_mask = masks[np.argmax(scores)]
            io = iou(max_mask, mask)
            fpr, fnr = fpr_fnr(max_mask, mask)
            for i, (mask_pred, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10,10))
                overlay = image.copy()
                overlay[mask_pred == True] = (0,255,0)
                overlay[mask == True] = (255,0,0)
                image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
                for point in input_points:
                    cv2.circle(image, (point[0], point[1]), 5, (0,0,255), -1)
                plt.imsave(f"../mask/{key}_mask_{i+1}_score_{score:.3f}.jpg", image)
                plt.close()
            io_list.append(io)
            fpr_list.append(fpr)
            fnr_list.append(fnr)
            print(io, fpr, fnr)
            pictures[key] = (io_list, fpr_list, fnr_list)
    return(io_list, fpr_list, fnr_list, pictures)


if __name__ == "__main__":
    io_list, fpr_list, fnr_list, pictures = all_combine_save_number()
    with open("Values.txt", "w") as file:
        for key,value in pictures.items():
            file.writelines(f"{key}: {value} \n")


