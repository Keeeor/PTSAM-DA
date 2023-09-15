import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm

from label_process import remove_small_block, show_anns, segment_boxes, remove_small_regions, show_box
from load_model import load_generator_hq_model, load_generator_model, load_predictor_hq_model, load_predictor_model


def seg_image_process(data_path, save_path, model_type, hq_model, edge):
    # 初始化generator模型 不需要提示框
    # if hq_model:
    #     generator = load_generator_hq_model(model_type)
    # else:
    #     generator = load_generator_model(model_type, device='cpu')

    # 初始化predictor模型 需要提示框
    if hq_model:
        predictor = load_predictor_hq_model(model_type)
    else:
        predictor = load_predictor_model(model_type)

    data_path = os.path.join(data_path)
    class_list = os.listdir(data_path)

    index = 0
    for cls in class_list:
        print('The number of completed categories :' + str(index))
        index += 1
        image_path = os.path.join(data_path, cls)
        new_image_path = os.path.join(save_path, cls)

        if not os.path.exists(new_image_path):
            os.makedirs(new_image_path)

        for name in tqdm(os.listdir(image_path)):
            image = Image.open(os.path.join(image_path, name))  # 加载图片
            if image.mode != "RGB":
                image.save(os.path.join(new_image_path, name))
                continue

            image = np.asarray(image)
            predictor.set_image(image)

            h, w, c = image.shape
            new_mask = np.zeros((h, w))

            #  图像分块策略
            boxes = segment_boxes(h, w, edge)

            for input_box in boxes:
                input_box = np.array(input_box)
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
                new_mask += mask

            # # segany模型生成整图mask
            # masks = generator.generate(image, False)
            # plt.imshow(image)
            # show_anns(masks)
            # plt.axis('off')
            # plt.show()
            #
            # for mask in masks:
            #     bbox = mask['bbox']
            #     boundary = mask['crop_box']
            #
            #     if bbox[0] > boundary[0] + 2 and bbox[1] > boundary[1] + 2 and \
            #             bbox[0] + bbox[2] < boundary[2] - 2 and bbox[1] + bbox[3] < boundary[3] - 2:
            #         sub_mask = mask['segmentation']
            #
            #         new_mask += sub_mask

            new_mask = new_mask.astype(np.uint8)
            rgb_mask = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2RGB)
            new_image = np.where(rgb_mask > 0, 0, image)

            # plt.imshow(image)
            # for box in boxes:
            #     show_box(box, plt.gca())
            # plt.axis('off')
            # plt.show()

            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(new_image_path, name), new_image)


def args_parser():
    parser = argparse.ArgumentParser()
    # /data/ImageNet100/train/
    parser.add_argument('data_path', type=str, help="Path of the original dataset.")
    parser.add_argument('save_path', type=str, help="Path of the new dataset.")
    parser.add_argument('--model_type', type=str, default='vit_h', help="vit_h,vit_l,vit_b")
    parser.add_argument('--hq_sam', type=bool, default=False, help="Whether to use hq_sam")
    parser.add_argument('--edge', type=float, default=0.2, help="The distance for edge cropping")
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    data_path = args.data_path
    save_path = args.save_path
    model_type = args.model_type
    hq_sam = args.hq_sam
    edge = args.edge
    if not os.path.exists(data_path):
        print("Dataset not found in " + data_path)
        return
    seg_image_process(data_path, save_path, model_type, hq_sam, edge)


if __name__ == '__main__':
    args = args_parser()
    main(args)
