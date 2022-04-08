import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import csv

from PIL import Image
from torchvision.utils import draw_bounding_boxes
from config import *
from time import sleep

def open_img(img_dir_path, img_path):
    full_img_path = os.path.join(img_dir_path, img_path)
    img = np.asarray(Image.open(full_img_path))
    return img


def img_to_mask(img_dir_path, img_path, coords):
    img = open_img(img_dir_path, img_path)
    img_width,img_height = img.shape[0],img.shape[1]
    mask = np.zeros(shape=(img_width, img_height))
    if coords != (-1, -1, -1, -1):
        ymin, xmin, ymax, xmax  = coords
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                mask[x,y] = 1
    return mask


def imgs_to_mask(labels_path, has_header=True):
    written_masks = []
    with open(labels_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        if has_header:
            next(csv_reader) # skip header
    
        for row in csv_reader:
            img_name, coords = row[0], list(map(int, map(float, row[1:])))
            curr_mask = img_to_mask(IMAGE_DATASET_PATH, img_name, coords)
            img = Image.fromarray(curr_mask)
            img.save(os.path.join(MASK_DATASET_PATH,img_name.split('.')[0]+'.tiff'))
            written_masks.append(img_name.split('.')[0])

    for img_name in os.listdir(IMAGE_DATASET_PATH):
        if img_name.split('.')[0] not in written_masks:
            curr_mask = img_to_mask(IMAGE_DATASET_PATH, img_name, (-1, -1, -1, -1))
            Image.fromarray(curr_mask).save(os.path.join(MASK_DATASET_PATH,img_name.split('.')[0]+'.tiff'))


if __name__ == '__main__':
    imgs_to_mask('data/train_bounding_boxes.csv')