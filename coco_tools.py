import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tqdm import tqdm

data_dir = "/data/coco/"
data_dir = 'D:/Program/segment-anything/data/COCOstuff/'

types = ['train','val']
for type in types:
    instance_path = os.path.join(data_dir, 'annotations', 'instances_' + type + '2017.json')
    coco = COCO(instance_path)
    # Create label directory if it does not exist
    label_dir = os.path.join(data_dir, "labels", type)
    os.makedirs(label_dir, exist_ok=True)

    imgs = coco.imgs
    imgAnns = coco.imgToAnns
    ann_list = imgAnns.keys()
    # Iterate over images
    for img_id in tqdm(imgs):
        # Get image ID and filename
        img = imgs[img_id]
        h = img['height']
        w = img['width']
        name = img['file_name'].replace('jpg', 'png')
        mask = np.zeros((h, w))

        if img_id in ann_list:
            anns = imgAnns[img_id]
            for ann in anns:
                category_id = ann['category_id']
                tmp = coco.annToMask(ann)
                mask = np.where(tmp > 0, category_id, mask)

        cv2.imwrite(os.path.join(label_dir, name), mask)
