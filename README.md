# Pre-Trained SAM as Data Augmentation for Image Segmentation

We propose a novel training-free method that utilizes the Pre-Trained Segment Anything Model (SAM) model as a Data Augmentation tool (PTSAM-DA), to generate the augmented samples for image segmentation.


###1.ImageNet分类图像增广
```python sam_generator.py data_path save_path --model_type --hq_sam --edge```

输入路径data_path和输出路径save_path都精确到到train/val一级。model_type从[vit_h,vit_l,vit_b]三选一，模型大小递减，效果也递减，默认vit_h。
hq_sam是布尔值，表示是否使用[hq_sam](https://arxiv.org/abs/2306.01567), 默认为False。edge为浮点数，
表示图像边缘擦除的范围， 默认值为0.2，表示擦除图像四周20%范围内的识别对象。

```python sam_generator.py /data/ImageNet100/val/ /data/ImageNet100_sam/val/ --model_type vit_h --hq_sam False --edge 0.2```
