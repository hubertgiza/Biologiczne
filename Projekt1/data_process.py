from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import csv


def open_img(img_dir_path, img_path):
    full_img_path = os.path.join(img_dir_path, img_path)
    img = read_image(full_img_path)
    return img


def img_to_mask(img_dir_path, img_path, coords):
    img = open_img(img_dir_path, img_path)
    img_height = img.shape[1] + 1
    img_width = img.shape[2] + 1
    mask = np.zeros(shape=(img_width, img_height))
    if coords != (-1, -1, -1, -1):
        xmin, ymin, xmax, ymax = coords
        for x in range(xmin, xmax+1):
            for y in range(ymin, ymax+1):
                mask[x,y] = 1
    return mask


def imgs_to_mask(imgs_dir_path, labels_path, has_header=True):
    masks = dict()
    with open(labels_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        if has_header:
            next(csv_reader) # skip header

        for row in csv_reader:
            img_path, coords = row[0], list(map(int, map(float, row[1:])))
            masks[img_path] = img_to_mask(imgs_dir_path, img_path, coords)

    for img_path in os.listdir(imgs_dir_path):
        if img_path not in masks:
            img_to_mask(imgs_dir_path, img_path, (-1, -1, -1, -1))
    return masks


def add_label(img_dir_path, img_path, coords):
    boxes = torch.IntTensor([coords])
    img = open_img(img_dir_path, img_path) 
    img = draw_bounding_boxes(img, boxes, colors='red')
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


def add_labels(imgs_dir_path, labels_path, has_header=True):
    with open(labels_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        if has_header:
            next(csv_reader) # skip header

        for row in csv_reader:
            img_path, coords = row[0], list(map(int, map(float, row[1:])))
            add_label(imgs_dir_path, img_path, coords)


if __name__ == '__main__':
    imgs_dir_path = './data/training_images/'
    labels_path = 'data/train_solution_bounding_boxes.csv'
    img_path = 'vid_4_1000.jpg'
    row = ['vid_4_1000.jpg','281.2590449','187.0350708','327.7279305','223.225547']
    img_path, coords = row[0], list(map(int, map(float, row[1:])))

    mask = img_to_mask(imgs_dir_path, img_path, coords)
    print(mask)
    add_label(imgs_dir_path, img_path, coords)
