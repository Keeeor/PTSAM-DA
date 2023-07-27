import argparse
import json
import os
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from tqdm import tqdm

from label_process import get_geo_bbox, remove_small_regions, show_box, remove_small_block
from load_model import load_generator_hq_model, load_generator_model

CLASSES = [1, 2]
CLASSES = [i for i in range(1, 255)]


def seg_image_process(root, images, new_images, model_type, hq_model):
    image_path = os.path.join(root, images)
    new_images_path = os.path.join(root, new_images)
    if not os.path.exists(new_images_path):
        os.mkdir(new_images_path)

    # 初始化segment-anything模型
    if hq_model:
        generator = load_generator_hq_model(model_type)
    else:
        generator = load_generator_model(model_type)

    for name in tqdm(os.listdir(image_path)):
        image = Image.open(os.path.join(image_path, name))  # 加载图片
        if image.mode != "RGB":
            continue
        image = np.asarray(image)
        size = image.shape
        new_mask = np.zeros(size[:2])
        # segany模型生成整图mask
        masks = generator.generate(image, False)

        for mask in masks:
            bbox = mask['bbox']
            boundary = mask['crop_box']

            if bbox[0] > boundary[0] and bbox[1] > boundary[1] and \
                    bbox[0] + bbox[2] < boundary[2] and bbox[1] + bbox[3] < boundary[3]:
                sub_mask = mask['segmentation']

                new_mask += sub_mask

        new_mask = remove_small_block(new_mask, 0.01)
        rgb_mask = cv2.cvtColor(new_mask,cv2.COLOR_GRAY2RGB)
        new_image = np.where(rgb_mask > 0, image, 0)

        cv2.imwrite(os.path.join(new_images_path, 'sam_' + name), new_image)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help="Path of the ori dataset.")
    parser.add_argument('--images', type=str, default='images', help="images dir name")
    parser.add_argument('--new_images', type=str, default='sam_images', help="labels dir name")
    parser.add_argument('--model_type', type=str, default='vit_h', help="vit_h,vit_l,vit_b")
    parser.add_argument('--hq_sam', type=bool, default=False, help="hq_sam")
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    data_path = args.data_path
    images = args.images
    new_images = args.new_images
    model_type = args.model_type
    hq_sam = args.hq_sam
    if not os.path.exists(data_path):
        print("Dataset not found in " + data_path)
        return
    seg_image_process(data_path, images, new_images, model_type, hq_sam)


if __name__ == '__main__':
    args = args_parser()
    main(args)
