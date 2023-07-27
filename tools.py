import os
import shutil
import random
from label_process import remove_small_regions, show_box, remove_small_block
from load_model import load_predictor_model, load_predictor_hq_model
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import cv2


def check_label(root, check_image, check_label):
    images_path = os.path.join(root, check_image)
    check_path = os.path.join(root, check_label)

    for img in tqdm(os.listdir(images_path)):
        check = img.replace('jpg', 'png')
        image = cv2.imread(os.path.join(images_path, img))
        label = cv2.imread(os.path.join(check_path, check))
        if image.shape == label.shape:
            continue
        else:
            print(img + ':' + str(image.shape) + "!=" + check + ':' + str(label.shape))


def copy_to_trainval(root):
    seg_any_mask_path = os.path.join(root, 'seg-any_mask')
    segmentation_path = os.path.join(root, 'segmentation')

    seg_any_train = os.path.join(segmentation_path, 'train', 'seg-any_labels')
    if not os.path.exists(seg_any_train):
        os.mkdir(seg_any_train)
    seg_any_val = os.path.join(segmentation_path, 'val', 'seg-any_labels')
    if not os.path.exists(seg_any_val):
        os.mkdir(seg_any_val)

    train_list = os.listdir(os.path.join(segmentation_path, 'train', 'labels'))
    val_list = os.listdir(os.path.join(segmentation_path, 'val', 'labels'))

    for mask in tqdm(os.listdir(seg_any_mask_path)):
        if mask in train_list:
            shutil.copyfile(os.path.join(seg_any_mask_path, mask), os.path.join(seg_any_train, mask))
        elif mask in val_list:
            shutil.copyfile(os.path.join(seg_any_mask_path, mask), os.path.join(seg_any_val, mask))
        else:
            print(mask + ' not found in trainval!')


def show_image(root=None):
    if root == None:
        root = 'D:/Desktop/show'
    fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(30, 42))
    fig.subplots_adjust(top=0.95)

    images = os.path.join(root, "images")
    # labels = os.path.join(root, "labels")
    inference = os.path.join(root, "inference")
    size = (700, 800)

    for i, img in enumerate(os.listdir(images)):
        image_path = os.path.join(images, img)
        # label_path = os.path.join(labels, img.replace('.jpg', '.png'))
        inference_path = os.path.join(inference, img)

        img_a = Image.open(image_path).resize(size)
        # img_b = Image.open(label_path).resize(size)
        img_c = Image.open(inference_path).resize(size)

        axs[i][0].imshow(img_a)
        # axs[i][1].imshow(img_b)
        axs[i][1].imshow(img_c)
        axs[i][0].axis('off')
        axs[i][1].axis('off')
        # axs[i][2].axis('off')
        if i == 0:
            axs[i, 0].set_title('Image', fontsize=40)
            # axs[i, 1].set_title('Label', fontsize=40)
            axs[i, 1].set_title('Inference', fontsize=40)

    # root = 'D:/Desktop/bad_show'
    # images = os.path.join(root, "images")
    # # labels = os.path.join(root, "labels")
    # inference = os.path.join(root, "inference")
    #
    # for i, img in enumerate(os.listdir(images)):
    #     image_path = os.path.join(images, img)
    #     # label_path = os.path.join(labels, img.replace('.jpg', '.png'))
    #     inference_path = os.path.join(inference, img)
    #
    #     img_a = Image.open(image_path).resize(size)
    #     # img_b = Image.open(label_path).resize(size)
    #     img_c = Image.open(inference_path).resize(size)
    #
    #     axs[i][2].imshow(img_a)
    #     # axs[i][1].imshow(img_b)
    #     axs[i][3].imshow(img_c)
    #     axs[i][2].axis('off')
    #     axs[i][3].axis('off')
    #     # axs[i][2].axis('off')
    #     if i == 0:
    #         axs[i, 2].set_title('Image', fontsize=40)
    #         # axs[i, 1].set_title('Label', fontsize=40)
    #         axs[i, 3].set_title('Inference', fontsize=40)
    # plt.tight_layout()
    # plt.show()


def create_geo_dataset(root):
    images_path = os.path.join(root, "compress_image")
    labels_path = os.path.join(root, "compress_mask")
    seg_path = os.path.join(root, "seg-any_mask")

    ori_dataset = os.path.join(root, "ori_dataset")
    ori_train_images = os.path.join(ori_dataset, "train", 'images')
    ori_train_labels = os.path.join(ori_dataset, "train", 'labels')
    ori_val_images = os.path.join(ori_dataset, "val", 'images')
    ori_val_labels = os.path.join(ori_dataset, "val", 'labels')
    seg_dataset = os.path.join(root, "seg_dataset")
    seg_train_images = os.path.join(seg_dataset, "train", 'images')
    seg_train_labels = os.path.join(seg_dataset, "train", 'labels')
    seg_val_images = os.path.join(seg_dataset, "val", 'images')
    seg_val_labels = os.path.join(seg_dataset, "val", 'labels')

    if not os.path.exists(ori_train_images):
        os.makedirs(ori_train_images)
    if not os.path.exists(ori_train_labels):
        os.makedirs(ori_train_labels)
    if not os.path.exists(ori_val_images):
        os.makedirs(ori_val_images)
    if not os.path.exists(ori_val_labels):
        os.makedirs(ori_val_labels)
    if not os.path.exists(seg_train_images):
        os.makedirs(seg_train_images)
    if not os.path.exists(seg_train_labels):
        os.makedirs(seg_train_labels)
    if not os.path.exists(seg_val_images):
        os.makedirs(seg_val_images)
    if not os.path.exists(seg_val_labels):
        os.makedirs(seg_val_labels)

    filelist = os.listdir(images_path)
    lenth = len(filelist)

    train_number = int(lenth * 0.8)
    train = random.sample(filelist, train_number)
    val = list(set(filelist) - set(train))

    # train
    for img in train:
        lab = img.replace('jpg', 'png')
        # images
        shutil.copyfile(os.path.join(images_path, img), os.path.join(ori_train_images, img))
        shutil.copyfile(os.path.join(images_path, img), os.path.join(seg_train_images, img))
        shutil.copyfile(os.path.join(images_path, img), os.path.join(seg_train_images, 'seg_' + img))

        # labels
        shutil.copyfile(os.path.join(labels_path, lab), os.path.join(ori_train_labels, lab))
        shutil.copyfile(os.path.join(labels_path, lab), os.path.join(seg_train_labels, lab))
        shutil.copyfile(os.path.join(seg_path, lab), os.path.join(seg_train_labels, 'seg_' + lab))

    # val
    for img in val:
        lab = img.replace('jpg', 'png')
        shutil.copyfile(os.path.join(images_path, img), os.path.join(ori_val_images, img))
        shutil.copyfile(os.path.join(images_path, img), os.path.join(seg_val_images, img))
        # shutil.copyfile(os.path.join(images_path, img), os.path.join(seg_val_images, 'seg_' + img))

        shutil.copyfile(os.path.join(labels_path, lab), os.path.join(ori_val_labels, lab))
        shutil.copyfile(os.path.join(labels_path, lab), os.path.join(seg_val_labels, lab))
        # shutil.copyfile(os.path.join(seg_path, lab), os.path.join(seg_val_labels, 'seg_' + lab))


def create_small_geo_dataset(root):
    images_path = os.path.join(root, "compress_image")
    labels_path = os.path.join(root, "compress_mask")
    seg_path = os.path.join(root, "seg-any_mask")

    ori_dataset = os.path.join(root, "ori_dataset")
    ori_train_images = os.path.join(ori_dataset, "train", 'images')
    ori_train_labels = os.path.join(ori_dataset, "train", 'labels')
    ori_val_images = os.path.join(ori_dataset, "val", 'images')
    ori_val_labels = os.path.join(ori_dataset, "val", 'labels')
    seg_dataset = os.path.join(root, "seg_dataset")
    seg_train_images = os.path.join(seg_dataset, "train", 'images')
    seg_train_labels = os.path.join(seg_dataset, "train", 'labels')
    seg_val_images = os.path.join(seg_dataset, "val", 'images')
    seg_val_labels = os.path.join(seg_dataset, "val", 'labels')

    if not os.path.exists(ori_train_images):
        os.makedirs(ori_train_images)
    if not os.path.exists(ori_train_labels):
        os.makedirs(ori_train_labels)
    if not os.path.exists(ori_val_images):
        os.makedirs(ori_val_images)
    if not os.path.exists(ori_val_labels):
        os.makedirs(ori_val_labels)
    if not os.path.exists(seg_train_images):
        os.makedirs(seg_train_images)
    if not os.path.exists(seg_train_labels):
        os.makedirs(seg_train_labels)
    if not os.path.exists(seg_val_images):
        os.makedirs(seg_val_images)
    if not os.path.exists(seg_val_labels):
        os.makedirs(seg_val_labels)

    farmland_list = os.listdir(os.path.join(root, 'small_farmland'))
    greenhouse_list = os.listdir(os.path.join(root, 'small_greenhouse'))

    farmland_train = random.sample(farmland_list, int(len(farmland_list) * 0.8))
    farmland_val = list(set(farmland_list) - set(farmland_train))

    greenhouse_train = random.sample(greenhouse_list, int(len(greenhouse_list) * 0.8))
    greenhouse_val = list(set(greenhouse_list) - set(greenhouse_train))

    train = farmland_train + greenhouse_train
    val = farmland_val + greenhouse_val

    # train
    for img in train:
        lab = img.replace('jpg', 'png')
        # images
        shutil.copyfile(os.path.join(images_path, img), os.path.join(ori_train_images, img))
        shutil.copyfile(os.path.join(images_path, img), os.path.join(seg_train_images, img))
        shutil.copyfile(os.path.join(images_path, img), os.path.join(seg_train_images, 'seg_' + img))

        # labels
        shutil.copyfile(os.path.join(labels_path, lab), os.path.join(ori_train_labels, lab))
        shutil.copyfile(os.path.join(labels_path, lab), os.path.join(seg_train_labels, lab))
        shutil.copyfile(os.path.join(seg_path, lab), os.path.join(seg_train_labels, 'seg_' + lab))

    # val
    for img in val:
        lab = img.replace('jpg', 'png')
        shutil.copyfile(os.path.join(images_path, img), os.path.join(ori_val_images, img))
        shutil.copyfile(os.path.join(images_path, img), os.path.join(seg_val_images, img))
        # shutil.copyfile(os.path.join(images_path, img), os.path.join(seg_val_images, 'seg_' + img))

        shutil.copyfile(os.path.join(labels_path, lab), os.path.join(ori_val_labels, lab))
        shutil.copyfile(os.path.join(labels_path, lab), os.path.join(seg_val_labels, lab))
        # shutil.copyfile(os.path.join(seg_path, lab), os.path.join(seg_val_labels, 'seg_' + lab))


def create_CoCo_dataset(root):
    images_path = os.path.join(root, 'images', 'train2017')
    seg_mask_path = os.path.join(root, 'coco_mask')
    ori_mask_path = os.path.join(root, 'seg_mask')

    dataset_path = os.path.join(root, 'coco_dataset')

    images_data = os.path.join(dataset_path, 'images')
    images_train = os.path.join(images_data, 'train')
    if not os.path.exists(images_train):
        os.makedirs(images_train)
    images_val = os.path.join(images_data, 'val')
    if not os.path.exists(images_val):
        os.makedirs(images_val)

    seg_maks_data = os.path.join(dataset_path, 'seg_labels')
    seg_mask_train = os.path.join(seg_maks_data, 'train')
    if not os.path.exists(seg_mask_train):
        os.makedirs(seg_mask_train)
    seg_mask_val = os.path.join(seg_maks_data, 'val')
    if not os.path.exists(seg_mask_val):
        os.makedirs(seg_mask_val)

    ori_maks_data = os.path.join(dataset_path, 'ori_labels')
    ori_mask_train = os.path.join(ori_maks_data, 'train')
    if not os.path.exists(ori_mask_train):
        os.makedirs(ori_mask_train)
    ori_mask_val = os.path.join(ori_maks_data, 'val')
    if not os.path.exists(ori_mask_val):
        os.makedirs(ori_mask_val)

    filelist = os.listdir(seg_mask_path)
    lenth = len(filelist)

    train_number = int(lenth * 0.7)

    train = random.sample(filelist, train_number)
    val = list(set(filelist) - set(train))

    for item in tqdm(filelist):
        if item in train:
            shutil.copyfile(os.path.join(seg_mask_path, item), os.path.join(seg_mask_train, item))
            shutil.copyfile(os.path.join(ori_mask_path, item), os.path.join(ori_mask_train, item))
            shutil.copyfile(os.path.join(images_path, item.replace('png', 'jpg')),
                            os.path.join(images_train, item.replace('png', 'jpg')))
        elif item in val:
            shutil.copyfile(os.path.join(seg_mask_path, item), os.path.join(seg_mask_val, item))
            shutil.copyfile(os.path.join(ori_mask_path, item), os.path.join(ori_mask_val, item))
            shutil.copyfile(os.path.join(images_path, item.replace('png', 'jpg')),
                            os.path.join(images_val, item.replace('png', 'jpg')))


# 不需要了，cityscape不需要把所有的图片都集中在一个文件夹下，白写了
def create_cityscape_dataset(root, extra=True):
    leftImg8bit_path = os.path.join(root, 'leftImg8bit')
    gtFine_path = os.path.join(root, 'gtFine')
    gtCoarse_path = os.path.join(root, 'gtCoarse')

    if extra:
        images_path = os.path.join(root, 'images_extra')
        labels_path = os.path.join(root, 'labels_extra')
    else:
        images_path = os.path.join(root, 'images')
        labels_path = os.path.join(root, 'labels')

    images_train = os.path.join(images_path, 'train')
    images_val = os.path.join(images_path, 'val')
    if not os.path.exists(images_train):
        os.makedirs(images_train)
    if not os.path.exists(images_val):
        os.makedirs(images_val)

    labels_train = os.path.join(labels_path, 'train')
    labels_val = os.path.join(labels_path, 'val')
    if not os.path.exists(labels_train):
        os.makedirs(labels_train)
    if not os.path.exists(labels_val):
        os.makedirs(labels_val)

    # 1.先移动train图片
    trian_path = os.path.join(leftImg8bit_path, 'train')
    for city in os.listdir(trian_path):
        print('Move images from ' + city)
        city_images_path = os.path.join(trian_path, city)
        for image in tqdm(os.listdir(city_images_path)):
            shutil.copyfile(os.path.join(city_images_path, image), os.path.join(images_train, image))

    # 2.再移动train_extra图片
    if extra:
        trian_path = os.path.join(leftImg8bit_path, 'train_extra')
        for city in os.listdir(trian_path):
            print('Move images from ' + city)
            city_images_path = os.path.join(trian_path, city)
            for image in tqdm(os.listdir(city_images_path)):
                shutil.copyfile(os.path.join(city_images_path, image), os.path.join(images_train, image))

    # 3.再移动val图片
    trian_path = os.path.join(leftImg8bit_path, 'val')
    for city in os.listdir(trian_path):
        print('Move images from ' + city)
        city_images_path = os.path.join(trian_path, city)
        for image in tqdm(os.listdir(city_images_path)):
            shutil.copyfile(os.path.join(city_images_path, image), os.path.join(images_val, image))

    # 4.先移动train标签
    val_path = os.path.join(gtFine_path, 'train')
    for city in os.listdir(val_path):
        print('Move labels from ' + city)
        city_labels_path = os.path.join(val_path, city)
        for label in tqdm(os.listdir(city_labels_path)):
            if label.endswith('_labelTrainIds.png'):
                shutil.copyfile(os.path.join(city_labels_path, label), os.path.join(labels_train, label))

    # 5.再移动train_extra标签
    if extra:
        val_path = os.path.join(gtCoarse_path, 'train_extra')
        for city in os.listdir(val_path):
            print('Move labels from ' + city)
            city_labels_path = os.path.join(val_path, city)
            for label in tqdm(os.listdir(city_labels_path)):
                if label.endswith('_labelTrainIds.png'):
                    shutil.copyfile(os.path.join(city_labels_path, label), os.path.join(labels_train, label))

    # 6.再移动val标签
    val_path = os.path.join(gtFine_path, 'val')
    for city in os.listdir(val_path):
        print('Move labels from ' + city)
        city_labels_path = os.path.join(val_path, city)
        for label in tqdm(os.listdir(city_labels_path)):
            if label.endswith('_labelTrainIds.png'):
                shutil.copyfile(os.path.join(city_labels_path, label), os.path.join(labels_val, label))

    print('Images train number: ' + str(len(os.listdir(images_train))))
    print('Labels train number: ' + str(len(os.listdir(labels_train))))
    print('Images val number: ' + str(len(os.listdir(images_val))))
    print('Labels val number: ' + str(len(os.listdir(labels_val))))


def create_ADE20K_dataset(root):
    images_path = os.path.join(root, 'images', 'training')
    annotations_path = os.path.join(root, 'annotations', 'training')
    seg_annotations_path = os.path.join(root, 'seg-annotations', 'training')

    new_images_path = os.path.join(root, 'new_images', 'training')
    new_annotations_path = os.path.join(root, 'new_annotations', 'training')
    if not os.path.exists(new_images_path):
        os.makedirs(new_images_path)
    if not os.path.exists(new_annotations_path):
        os.makedirs(new_annotations_path)

    images_list = os.listdir(images_path)
    annotations_list = os.listdir(annotations_path)
    seg_annotations_list = os.listdir(seg_annotations_path)

    for annotation in annotations_list:
        shutil.copyfile(os.path.join(annotations_path, annotation), os.path.join(new_annotations_path, annotation))

    for seg_annotation in seg_annotations_list:
        shutil.copyfile(os.path.join(seg_annotations_path, seg_annotation),
                        os.path.join(new_annotations_path, seg_annotation))

    for image in images_list:
        shutil.copyfile(os.path.join(images_path, image), os.path.join(new_images_path, image))
        shutil.copyfile(os.path.join(images_path, image), os.path.join(new_images_path, 'seg_' + image))


def check_label(root):
    plt.figure(figsize=(10, 10))
    for name in tqdm(os.listdir(root)):
        label_path = os.path.join(root, name)
        label_img = Image.open(label_path)
        label = np.asarray(label_img)
        person = np.sum(label == 255)
        if person / label.size < 0.1:
            os.remove(label_path)


def check_VOC_label(root):
    for label in tqdm(os.listdir(root)):
        label_path = os.path.join(root, label)
        label_img = Image.open(label_path)
        color_map = label_img.palette.colors
        img = np.asarray(label_img)
        # classes = np.unique(image)
        plt.figure(figsize=(10, 10))
        plt.title('mask')
        plt.imshow(label_img)
        plt.show()

        # if len(classes) == 2 and 0 in classes and 1 in classes:
        #     continue
        # else:
        #     print(label + ":" + str(classes))


def seg_laebl_play():
    hq_model = False
    root = 'D:\Desktop\classes_08\merge_house\compress_0.1_images_1\idea'
    image_path = os.path.join(root, 'compress_image')
    label_path = os.path.join(root, 'compress_mask')
    seg_label_path = os.path.join(root, "seg_mask")
    if not os.path.exists(seg_label_path):
        os.mkdir(seg_label_path)

    # 初始化segment-anything模型
    if hq_model:
        predictor = load_predictor_hq_model('vit_h')
    else:
        predictor = load_predictor_model('vit_h')

    label = Image.open(os.path.join(label_path, 'image_1.png'))  # 加载原始标注
    label = np.asarray(label)
    classes = np.unique(label)

    image = Image.open(os.path.join(image_path, 'image_1.jpg'))  # 加载图片
    image = np.asarray(image)

    seg_label = label.copy()

    # seg_label = np.where(seg_label == 2, seg_label, 0)

    fig, ax = plt.subplots(1, 4, figsize=(10, 3))
    ax[0].imshow(image)
    ax[0].axis('off')

    ax[1].imshow(image)
    ax[1].imshow(seg_label, alpha=0.5)
    ax[1].axis('off')

    # segany模型加载图片
    predictor.set_image(image)

    for class_id in classes:
        if class_id not in [1,2]:
            continue
        obj_mask = np.where(seg_label == class_id, 1, 0)

        show_mask = obj_mask.copy()
        if class_id ==1:
            show_mask = np.where(show_mask == 1, 128, show_mask)
        elif class_id ==2:
            show_mask = np.where(show_mask == 1, 255, show_mask)
        show_mask = show_mask.astype(np.uint8)

        cv2.imwrite(os.path.join(seg_label_path, str(class_id) + "_labelclass.png"), show_mask)




        # 根据mask计算框并使用seg识别
        obj_mask = obj_mask.astype(np.uint8)
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        seg_mask = np.zeros_like(label)
        box_mask = np.zeros_like(label)
        box_mask = cv2.cvtColor(box_mask, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            input_box = np.array([x, y, x + w, y + h])

            line = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
            cv2.polylines(box_mask, [line], True, (0, 0, 255), 5)

            cv2.imwrite(os.path.join(seg_label_path, str(class_id)+"_boxesclass.png"), box_mask)

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
            ax[2].imshow(show_mask, alpha=0.5)
            ax[2].axis('off')

        if class_id ==1:
            sam_mask = np.where(seg_mask == 1, 128, seg_mask)
        elif class_id ==2:
            sam_mask = np.where(seg_mask == 2, 255, seg_mask)
        sam_mask = sam_mask.astype(np.uint8)
        cv2.imwrite(os.path.join(seg_label_path, str(class_id) + "_samclass.png"), sam_mask)
        # 借助原始标签修复识别结果
        xor_mask = np.where(obj_mask == seg_mask, 0, 1)

        # 避免识别反的情况
        if (np.sum(xor_mask == 1) / obj_mask.size) > 0.5:
            continue

        xor_mask = remove_small_block(xor_mask, 0.01)

        seg_mask = np.where(xor_mask > 0, class_id, seg_mask)  # 修补标注
        seg_label = np.where(label == class_id, 0, seg_label)  # 清空原本标注
        seg_label = np.where(seg_mask > 0, class_id, seg_label)  # 更新标注

    ax[3].imshow(image)
    ax[3].imshow(seg_label, alpha=0.5)
    ax[3].axis('off')
    plt.tight_layout()
    plt.show()

    # cv2.imwrite(os.path.join(seg_label_path, name), seg_label)


if __name__ == '__main__':
    seg_laebl_play()
    # root = '/data/08/compress_0.1_images_1/'
    # create_small_geo_dataset(root)

    # root = 'D:/Program/segment-anything/data/COCOstuff/coco_mask'
    # check_label(root)

    # cocodataset
    # root = '/data/coco/'
    # create_CoCo_dataset(root)

    # root = 'data/VOC2012/SegmentationClass'
    # check_VOC_label(root)

    # create_cityscape_dataset(root, False)
    # create_cityscape_dataset(root, True)

    # check_label(root, 'images', 'labels')
    # copy_to_trainval(root)
    # show_image()
