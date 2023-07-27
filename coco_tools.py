import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tqdm import tqdm

data_dir = "/data/coco/"

types = ['train','val']
for type in types:
    instance_path = os.path.join(data_dir, 'annotations', 'instances_' + type + '2017.json')
    coco = COCO(instance_path)
    # Create label directory if it does not exist
    label_dir = os.path.join(data_dir, "labels", type)
    os.makedirs(label_dir, exist_ok=True)

    imgs = coco.imgs
    imgAnns = coco.imgToAnns

    # Iterate over images
    for image_id in tqdm(imgAnns):
        # Get image ID and filename
        img = imgs[image_id]
        h = img['height']
        w = img['width']
        name = img['file_name'].replace('jpg', 'png')
        mask = np.zeros((h, w))
        anns = imgAnns[image_id]

        for ann in anns:
            category_id = ann['category_id']
            tmp = coco.annToMask(ann)
            mask = np.where(tmp > 0, category_id, mask)

        cv2.imwrite(os.path.join(label_dir, name), mask)
