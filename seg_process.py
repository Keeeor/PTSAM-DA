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
from load_model import load_predictor_model, load_predictor_hq_model

CLASSES = [1, 2]
CLASSES = [i for i in range(255)]


def seg_laebl_process(root, images, labels, model_type, hq_model):
    image_path = os.path.join(root, images)
    label_path = os.path.join(root, labels)
    segany_label_path = os.path.join(root, "seg_mask")
    if not os.path.exists(segany_label_path):
        os.mkdir(segany_label_path)

    # 初始化segment-anything模型
    if hq_model:
        predictor = load_predictor_hq_model(model_type)
    else:
        predictor = load_predictor_model(model_type)

    for name in tqdm(os.listdir(label_path)):
        label = Image.open(os.path.join(label_path, name))  # 加载原始标注
        label = np.asarray(label)
        classes = np.unique(label)

        image = Image.open(os.path.join(image_path, name.replace('png', 'jpg')))  # 加载图片
        image = np.asarray(image)

        seg_label = label.copy()

        # seg_label = np.where(seg_label == 2, seg_label, 0)

        # fig, ax = plt.subplots(1, 4, figsize=(10, 3))
        # ax[0].imshow(image)
        # ax[0].axis('off')

        # ax[1].imshow(image)
        # ax[1].imshow(seg_label, alpha=0.5)
        # ax[1].axis('off')

        # segany模型加载图片
        predictor.set_image(image)

        for class_id in classes:
            if class_id not in CLASSES:
                continue
            obj_mask = np.where(seg_label == class_id, 1, 0)

            # 根据mask计算框并使用seg识别
            obj_mask = obj_mask.astype(np.uint8)
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            seg_mask = np.zeros_like(label)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                input_box = np.array([x, y, x + w, y + h])

                # show_box(input_box, plt.gca())
                if hq_model:
                    masks, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                        hq_token_only=False,
                    )
                else:
                    masks, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )

                mask = masks[0]
                mask, _ = remove_small_regions(mask, 1000, "holes")
                mask, _ = remove_small_regions(mask, 1000, "islands")
                seg_mask = np.where(mask, class_id, seg_mask)

                # ax[2].imshow(image)
                # ax[2].imshow(seg_mask, alpha=0.5)
                # ax[2].axis('off')

            # 借助原始标签修复识别结果
            xor_mask = np.where(obj_mask == seg_mask, 0, 1)

            # 避免识别反的情况
            if (np.sum(xor_mask == 1) / obj_mask.size) > 0.5:
                continue

            xor_mask = remove_small_block(xor_mask, 0.05)

            seg_mask = np.where(xor_mask > 0, class_id, seg_mask)  # 修补标注
            seg_label = np.where(label == class_id, 0, seg_label)  # 清空原本标注
            seg_label = np.where(seg_mask > 0, class_id, seg_label)  # 更新标注

        # ax[3].imshow(image)
        # ax[3].imshow(seg_label, alpha=0.5)
        # ax[3].axis('off')
        # plt.tight_layout()
        # plt.show()

        cv2.imwrite(os.path.join(segany_label_path, name), seg_label)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help="Path of the ori dataset.")
    parser.add_argument('--images', type=str, default='compress_image', help="images dir name")
    parser.add_argument('--labels', type=str, default='compress_mask', help="labels dir name")
    parser.add_argument('--model_type', type=str, default='vit_h', help="vit_h,vit_l,vit_b")
    parser.add_argument('--hq_sam', type=bool, default=False, help="hq_sam")
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    data_path = args.data_path
    images = args.images
    labels = args.labels
    model_type = args.model_type
    hq_sam = args.hq_sam
    if not os.path.exists(data_path):
        print("Dataset not found in " + data_path)
        return
    seg_laebl_process(data_path, images, labels, model_type, hq_sam)


if __name__ == '__main__':
    args = args_parser()
    main(args)
