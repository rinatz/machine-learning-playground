#!/usr/bin/env python

from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from object_detection.utils import ops
from object_detection.utils.label_map_util import create_category_index_from_labelmap
from object_detection.utils.visualization_utils import (
    visualize_boxes_and_labels_on_image_array,
)
from rich import print
import typer

MODEL_URLS = {
    "centernet_hg104_512x512_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz",
    "centernet_hg104_512x512_kpts_coco17_tpu-32": "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz",
    "centernet_hg104_1024x1024_coco17_tpu-32": "http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz",
    "centernet_hg104_1024x1024_kpts_coco17_tpu-32": "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_kpts_coco17_tpu-32.tar.gz",
    "centernet_resnet50_v1_fpn_512x512_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz",
    "centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8.tar.gz",
    "centernet_resnet101_v1_fpn_512x512_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz",
    "centernet_resnet50_v2_512x512_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz",
    "centernet_resnet50_v2_512x512_kpts_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_kpts_coco17_tpu-8.tar.gz",
    "efficientdet_d0_coco17_tpu-32": "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz",
    "efficientdet_d1_coco17_tpu-32": "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz",
    "efficientdet_d2_coco17_tpu-32": "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz",
    "efficientdet_d3_coco17_tpu-32": "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz",
    "efficientdet_d4_coco17_tpu-32": "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz",
    "efficientdet_d5_coco17_tpu-32": "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz",
    "efficientdet_d6_coco17_tpu-32": "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz",
    "efficientdet_d7_coco17_tpu-32": "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz",
    "ssd_mobilenet_v2_320x320_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz",
    "ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz",
    "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz",
    "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz",
    "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz",
    "ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz",
    "ssd_resnet101_v1_fpn_640x640_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz",
    "ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz",
    "ssd_resnet152_v1_fpn_640x640_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz",
    "ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz",
    "faster_rcnn_resnet50_v1_640x640_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz",
    "faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz",
    "faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.tar.gz",
    "faster_rcnn_resnet101_v1_640x640_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz",
    "faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz",
    "faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz",
    "faster_rcnn_resnet152_v1_640x640_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz",
    "faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz",
    "faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.tar.gz",
    "faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz",
    "faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz",
    "mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8": "http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz",
    "extremenet": "http://download.tensorflow.org/models/object_detection/tf2/20200711/extremenet.tar.gz",
}


def load_model(model_name):
    url = MODEL_URLS[model_name]

    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=url,
        untar=True,
    )

    model_dir = Path(model_dir).joinpath("saved_model")
    model = tf.saved_model.load(str(model_dir))

    return model


def load_category_index(label_map):
    label_map_path = f"models/research/object_detection/data/{label_map}.pbtxt"

    category_index = create_category_index_from_labelmap(
        label_map_path,
        use_display_name=True,
    )

    return category_index


def predict(model, image):
    image = np.asarray(image)
    x = tf.convert_to_tensor(image)
    x = x[tf.newaxis, ...]

    model_fn = model.signatures["serving_default"]
    y = model_fn(x)

    num_detections = int(y.pop("num_detections"))

    prediction = {key: val[0, :num_detections].numpy() for key, val in y.items()}
    prediction["num_detections"] = num_detections
    prediction["detection_classes"] = prediction["detection_classes"].astype(np.int64)

    if "detection_masks" in prediction:
        detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
            prediction["detection_masks"],
            prediction["detection_boxes"],
            image.shape[0],
            image.shape[1],
        )
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)

        prediction["detection_masks_reframed"] = detection_masks_reframed.numpy()

    return prediction


def main(
    image_path: str = typer.Argument(...),
    output: str = typer.Option("output.jpg"),
    model_name: str = typer.Option(
        "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8", "--model", "-m"
    ),
):
    model = load_model(model_name)
    category_index = load_category_index("mscoco_label_map")

    image = np.array(Image.open(image_path))
    prediction = predict(model, image)

    print(prediction)

    visualize_boxes_and_labels_on_image_array(
        image,
        prediction["detection_boxes"],
        prediction["detection_classes"],
        prediction["detection_scores"],
        category_index,
        instance_masks=prediction.get("detection_masks_reframed", None),
        use_normalized_coordinates=True,
        line_thickness=8,
    )

    Image.fromarray(image).save(output)


if __name__ == "__main__":
    typer.run(main)
