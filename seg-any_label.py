import json
import os
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from tqdm import tqdm

from label_process import get_geo_bbox, remove_small_regions, show_box, get_Cityscapes_bbox, remove_small_block
from load_model import load_predictor_model, load_predictor_hq_model
from pycocotools.coco import COCO
# from shapely import Polygon
from segment_anything import predictor
from tools import create_CoCo_dataset, create_ADE20K_dataset, create_geo_dataset

VIT_H = 'vit_h'
VIT_L = 'vit_l'
VIT_B = 'vit_b'
DATASET_TYPE = ['geo', 'Cityscapes', 'CoCo', 'ADE20K']

GEO_CLASSES = [1, 2]
Cityscapes_classes = {'unlabeled': 0, 'ego vehicle': 1, 'rectification border': 2, 'out of roi': 3, 'static': 4,
                      'dynamic': 5, 'ground': 6, 'road': 7, 'sidewalk': 8, 'parking': 9, 'rail track': 10,
                      'building': 11, 'wall': 12, 'fence': 13, 'guard rail': 14, 'bridge': 15, 'tunnel': 16, 'pole': 17,
                      'polegroup': 18, 'traffic light': 19, 'traffic sign': 20, 'vegetation': 21, 'terrain': 22,
                      'sky': 23, 'person': 24, 'rider': 25, 'car': 26, 'cargroup': 26, 'truck': 27, 'bus': 28,
                      'caravan': 29, 'trailer': 30, 'train': 31, 'motorcycle': 32, 'bicycle': 33}
Cityscapes_available_classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                                'traffic sign', 'vegetation', 'terrain', 'person', 'rider', 'car', 'cargroup',
                                'truck', 'bus', 'train', 'motorcycle', 'bicycle']
Cityscapes_trainId = {'road': 0,
                      'sidewalk': 1,
                      'building': 2,
                      'wall': 3,
                      'fence': 4,
                      'pole': 5,
                      'traffic light': 6,
                      'traffic sign': 7,
                      'vegetation': 8,
                      'terrain': 9,
                      'sky': 10,
                      'person': 11,
                      'rider': 12,
                      'car': 13,
                      'cargroup': 13,
                      'truck': 14,
                      'bus': 15,
                      'train': 16,
                      'motorcycle': 17,
                      'bicycle': 18}
Cityscapes_instance_classes = ['person', 'rider', 'car', 'cargroup', 'truck', 'bus', 'caravan', 'trailer', 'train',
                               'motorcycle', 'bicycle']

COCOstuff_mapping = {
    1: 255,  # person
    # 2: 2,  # bicycle
    # 3: 3,  # car
    # 4: 4,  # motorcycle
    # 6: 5,  # bus
    # 7: 6,  # train
    # 8: 7,  # truck
}

ADE20K_classes = [42, 53, 54, 61, 68, 69, 84, 85, 87, 88, 92, 93, 94, 96, 99, 101, 102, 104, 107, 109, 111, 113, 124,
                  126, 132, 135, 136, 137, 138, 139, 144, 149, 150]

ADE20K_classes = [31, 35, 61, 63, 64, 68, 69, 70, 85, 89, 98, 99, 104, 105, 110, 111, 114, 119, 120, 121, 125,
                  127, 131, 133, 144, 150]

ADE20K_classes = list(range(1, 151))

ADE20K_classes = [27, 30, 45, 47, 64, 82, 94, 101, 103, 104, 106, 114, 117, 120, 144]


def create_geo_segany_laebl(root, model_type, hq_model):
    image_path = os.path.join(root, "compress_image")
    ori_mask_path = os.path.join(root, "compress_mask")
    segany_label_path = os.path.join(root, "seg-any_mask")
    if not os.path.exists(segany_label_path):
        os.mkdir(segany_label_path)

    # 初始化segment-anything模型
    if hq_model:
        predictor = load_predictor_hq_model(model_type)
    else:
        predictor = load_predictor_model(model_type)

    for name in tqdm(os.listdir(ori_mask_path)):
        label = Image.open(os.path.join(ori_mask_path, name))  # 加载原始标注
        label = np.asarray(label)
        classes = np.unique(label)

        image = Image.open(os.path.join(image_path, name.replace('png', 'jpg')))  # 加载图片
        image = np.asarray(image)

        seg_label = label.copy()

        # seg_label = np.where(seg_label == 2, seg_label, 0)

        fig, ax = plt.subplots(1, 5, figsize=(10, 3))
        ax[0].imshow(image)
        ax[0].axis('off')

        ax[1].imshow(image)
        ax[1].imshow(seg_label, alpha=0.5)
        ax[1].axis('off')

        # segany模型加载图片
        predictor.set_image(image)

        for class_id in classes:
            if class_id not in GEO_CLASSES:
                continue
            ori_mask = np.where(label == class_id, 1, 0)

            # 根据mask计算框并使用seg识别
            obj_mask = ori_mask.astype(np.uint8)
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            segany_mask = np.zeros_like(label)

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
                segany_mask = np.where(mask, class_id, segany_mask)

            ax[2].imshow(image)
            ax[2].imshow(segany_mask, alpha=0.5)
            ax[2].axis('off')

            # 借助原始标签修复识别结果
            xor_mask = np.where(ori_mask == segany_mask, 0, 1)

            # 避免识别反的情况
            if (np.sum(xor_mask == 1) / ori_mask.size) > 0.5:
                continue

            xor_mask = remove_small_block(xor_mask, 0.05)
            ax[3].imshow(xor_mask, alpha=0.5)
            ax[3].axis('off')

            segany_mask = np.where(xor_mask > 0, class_id, segany_mask)  # 修补标注
            seg_label = np.where(label == class_id, 0, seg_label)  # 清空原本标注
            seg_label = np.where(segany_mask > 0, class_id, seg_label)  # 更新标注

        ax[4].imshow(image)
        ax[4].imshow(seg_label, alpha=0.5)
        ax[4].axis('off')
        plt.tight_layout()
        plt.show()

        cv2.imwrite(os.path.join(segany_label_path, name), seg_label)


def create_Cityscapes_segany_laebl(root, model_type=VIT_B):
    city_image_path = os.path.join(root, "leftImg8bit", "train_extra")
    city_label_path = os.path.join(root, "gtCoarse", "train_extra")
    # 初始化segment-anything模型
    predictor = load_predictor_model(model_type)

    for city in os.listdir(city_image_path):
        print("processing city:" + city)
        image_path = os.path.join(city_image_path, city)
        json_label_path = os.path.join(city_label_path, city)

        for img in tqdm(os.listdir(image_path)):
            image = cv2.imread(os.path.join(image_path, img))  # 加载图片
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            # 加载原始标注
            # ori_mask = cv2.imread(
            #     os.path.join(json_label_path, img.replace('_leftImg8bit.png', '_gtCoarse_labelIds.png')),
            #     cv2.IMREAD_GRAYSCALE)
            # 实例标注
            # ori_mask = cv2.imread(
            #     os.path.join(json_label_path, img.replace('_leftImg8bit.png', '_gtCoarse_instanceIds.png')),
            #     cv2.IMREAD_GRAYSCALE)

            # plt.figure(figsize=(10, 10))
            # plt.title('ori-mask')
            # plt.imshow(ori_mask)
            # plt.show()

            # 加载json标注
            json_name = img.replace('_leftImg8bit.png', '_gtCoarse_polygons.json')
            with open(os.path.join(json_label_path, json_name), "r") as f:
                jn = json.load(f)
            w = jn['imgWidth']
            h = jn['imgHeight']
            objects = jn['objects']

            segany_mask = np.zeros((h, w)) + 255
            # segany_mask = ori_mask

            if len(objects) is None:
                continue

            # plt.figure(figsize=(10, 10))
            for obj in objects:
                class_name = obj['label']
                if Cityscapes_trainId.__contains__(class_name):
                    class_id = Cityscapes_trainId[class_name]
                else:
                    continue
                bbox = get_Cityscapes_bbox(obj, w, h, 20)
                # 添加框方便观察补全效果
                # show_box(bbox, plt.gca())

                input_box = np.array(bbox)

                mask, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )

                mask, _ = remove_small_regions(mask[0], 2000, "holes")
                mask, _ = remove_small_regions(mask, 1000, "islands")

                segany_mask = np.where(mask, class_id, segany_mask)

            # plt.figure(figsize=(10, 10))
            # plt.title('seg-any')
            # plt.imshow(segany_mask)
            # plt.show()

            cv2.imwrite(os.path.join(json_label_path, img.replace("_leftImg8bit.png", "_gtCoarse_labelTrainIds.png")),
                        segany_mask)


def create_COCOstuff_segany_laebl(root, model_type=VIT_B, get_ori_mask=True, get_seg_mask=True):
    image_path = os.path.join(root, 'images', 'train2017')
    instance_path = os.path.join(root, 'annotations', 'instances_train2017.json')
    coco = COCO(instance_path)

    label_path = os.path.join(root, 'seg_mask')
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    coco_mask_path = os.path.join(root, 'coco_mask')
    if not os.path.exists(coco_mask_path):
        os.mkdir(coco_mask_path)

    # 初始化segment-anything模型
    predictor = load_predictor_model(model_type)

    # with open(instance_path, "r") as f:
    #     instances = json.load(f)

    images = coco.imgs
    images_anns = coco.imgToAnns

    for image_id in tqdm(images.keys()):
        image = images[image_id]
        image_name = image['file_name']

        height = image['height']
        width = image['width']
        anns = images_anns[image_id]

        image = Image.open(os.path.join(image_path, image_name))  # 加载图片
        # 图片质量缺陷
        if image.mode != "RGB":
            continue
        image = np.asarray(image)

        # 加载图片
        predictor.set_image(image)

        segany_mask = np.zeros((height, width), dtype=np.uint8)
        ori_mask = np.zeros((height, width), dtype=np.uint8)

        # 图像是否包含指定类别
        is_empty = True

        for ann in anns:
            category_id = ann['category_id']
            if category_id in COCOstuff_mapping.keys():
                class_id = COCOstuff_mapping[category_id]
                is_empty = False
            else:
                continue

            if not ann['iscrowd']:
                if get_ori_mask:
                    obj_mask = coco.annToMask(ann)
                    ori_mask = np.where(obj_mask == 1, class_id, ori_mask)

                if get_seg_mask:
                    [x, y, w, h] = ann['bbox']  # x,y,w,h
                    bbox = [x, y, x + w, y + h]
                    input_box = np.array(bbox)
                    mask, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )

                    mask = mask[0]
                    mask, _ = remove_small_regions(mask, 1000, "holes")
                    mask, _ = remove_small_regions(mask, 1000, "islands")
                    segany_mask = np.where(mask, class_id, segany_mask)
            else:
                continue

        if not is_empty:
            # 占比太少的人不要
            person = np.sum(ori_mask == class_id)
            if person / ori_mask.size < 0.1:
                continue
            if get_seg_mask:
                # 借助原始标签修复识别结果
                xor_mask = np.where(ori_mask == segany_mask, 0, 1)
                # 避免识别反的情况
                if (np.sum(xor_mask == 1) / ori_mask.size) > 0.5:
                    continue
                xor_mask, _ = remove_small_regions(xor_mask, 1000, "islands")
                segany_mask = np.where(xor_mask > 0, class_id, segany_mask)
                cv2.imwrite(os.path.join(label_path, image_name.replace(".jpg", ".png")), segany_mask)
            if get_ori_mask:
                cv2.imwrite(os.path.join(coco_mask_path, image_name.replace(".jpg", ".png")), ori_mask)


def create_ADE20K_segany_laebl(root, model_type):
    images_path = os.path.join(root, 'images')
    lables_path = os.path.join(root, 'annotations')
    seg_path = os.path.join(root, 'seg-annotations')

    image_train_path = os.path.join(images_path, 'training')
    lable_train_path = os.path.join(lables_path, 'training')
    seg_train_path = os.path.join(seg_path, 'training')
    if not os.path.exists(seg_train_path):
        os.makedirs(seg_train_path)

    # 初始化segment-anything模型
    predictor = load_predictor_model(model_type)

    count = 0

    for name in tqdm(os.listdir(lable_train_path)):
        new_name = 'seg_' + name
        # if name in os.listdir(seg_train_path):
        #     continue
        label = Image.open(os.path.join(lable_train_path, name))  # 加载原始标注
        label = np.asarray(label)
        classes = np.unique(label)

        # 不存在需要修改类别
        if not set(classes).intersection(set(ADE20K_classes)):
            shutil.copyfile(os.path.join(lable_train_path, name), os.path.join(seg_train_path, new_name))
            continue
        count += 1
        image = Image.open(os.path.join(image_train_path, name.replace('png', 'jpg')))  # 加载图片
        # 图片质量缺陷
        if image.mode != "RGB":
            shutil.copyfile(os.path.join(lable_train_path, name), os.path.join(seg_train_path, new_name))
            continue
        image = np.asarray(image)

        seg_label = label.copy()
        # plt.figure(figsize=(10, 10))
        # plt.imshow(seg_label)
        # plt.show()
        # plt.close()

        # segany模型加载图片
        predictor.set_image(image)

        for class_id in classes:
            if class_id not in ADE20K_classes:
                continue
            ori_mask = np.where(label == class_id, 1, 0)

            # 根据mask计算框并使用seg识别
            bboxes = []
            obj_mask = ori_mask.astype(np.uint8)
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append([x, y, x + w, y + h])

            input_boxes = torch.tensor(bboxes, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            segany_mask = np.zeros_like(label)
            for mask in masks:
                mask = mask.cpu().numpy()[0]
                mask, _ = remove_small_regions(mask, 1000, "holes")
                mask, _ = remove_small_regions(mask, 100, "islands")
                segany_mask = np.where(mask, class_id, 0)

            # 借助原始标签修复识别结果
            xor_mask = np.where(ori_mask == segany_mask, 0, 1)

            # 避免识别反的情况
            if (np.sum(xor_mask == 1) / ori_mask.size) > 0.5:
                continue
            xor_mask, _ = remove_small_regions(xor_mask, 100, "islands")

            segany_mask = np.where(xor_mask > 0, class_id, segany_mask)
            seg_label = np.where(label == class_id, 0, seg_label)
            seg_label = np.where(segany_mask > 0, class_id, seg_label)

            # plt.figure(figsize=(10, 10))
            # plt.imshow(seg_label)
            # plt.show()
            # plt.close()

        cv2.imwrite(os.path.join(seg_train_path, new_name), seg_label)

    print("共修改图片标注数量：" + str(count))


def create_segany_label(root, dataset_type, model_type, hq_model):
    assert dataset_type in DATASET_TYPE, 'DataSet Type Error'
    if dataset_type is DATASET_TYPE[0]:
        create_geo_segany_laebl(root, model_type, hq_model)
    elif dataset_type is DATASET_TYPE[1]:
        create_Cityscapes_segany_laebl(root, model_type)
    elif dataset_type is DATASET_TYPE[2]:
        create_COCOstuff_segany_laebl(root, model_type)
    elif dataset_type is DATASET_TYPE[3]:
        create_ADE20K_segany_laebl(root, model_type)


if __name__ == '__main__':
    root = 'D:\Desktop\classes_08\merge_house\compress_0.1_images_1\idea'
    # root = '/data/08/compress_0.1_images_1'
    # geo, Cityscapes, CoCOo, ADE20K
    dataset_type = "geo"
    # VIT_H(BIG),VIT_L，VIT_B(SMALL)
    hq_model = False
    model_type = VIT_H
    create_segany_label(root, dataset_type, model_type, hq_model)
    # create_geo_dataset(root)
