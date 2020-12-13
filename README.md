# Playground for TensorFlow Model Garden

This repository is a playground for [TensorFlow Model Garden]

[TensorFlow Model Garden]: https://github.com/tensorflow/models

## Requirements

- Python >=3.7
- Poetry
- Makefile

## Install

```shell
$ make install
```

## Usage

```shell
$ poetry run python detect.py IMAGE [--model MODEL_NAME]
```

You can pass below model names to `--model`.

- `centernet_hg104_512x512_coco17_tpu-8`
- `centernet_hg104_512x512_kpts_coco17_tpu-32`
- `centernet_hg104_1024x1024_coco17_tpu-32`
- `centernet_hg104_1024x1024_kpts_coco17_tpu-32`
- `centernet_resnet50_v1_fpn_512x512_coco17_tpu-8`
- `centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8`
- `centernet_resnet101_v1_fpn_512x512_coco17_tpu-8`
- `centernet_resnet50_v2_512x512_coco17_tpu-8`
- `centernet_resnet50_v2_512x512_kpts_coco17_tpu-8`
- `efficientdet_d0_coco17_tpu-32`
- `efficientdet_d1_coco17_tpu-32`
- `efficientdet_d2_coco17_tpu-32`
- `efficientdet_d3_coco17_tpu-32`
- `efficientdet_d4_coco17_tpu-32`
- `efficientdet_d5_coco17_tpu-32`
- `efficientdet_d6_coco17_tpu-32`
- `efficientdet_d7_coco17_tpu-32`
- `ssd_mobilenet_v2_320x320_coco17_tpu-8`
- `ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8`
- `ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8`
- `ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8`
- `ssd_resnet50_v1_fpn_640x640_coco17_tpu-8`
- `ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8`
- `ssd_resnet101_v1_fpn_640x640_coco17_tpu-8`
- `ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8`
- `ssd_resnet152_v1_fpn_640x640_coco17_tpu-8`
- `ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8`
- `faster_rcnn_resnet50_v1_640x640_coco17_tpu-8`
- `faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8`
- `faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8`
- `faster_rcnn_resnet101_v1_640x640_coco17_tpu-8`
- `faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8`
- `faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8`
- `faster_rcnn_resnet152_v1_640x640_coco17_tpu-8`
- `faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8`
- `faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8`
- `faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8`
- `faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8`
- `mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8`
- `extremenet`
