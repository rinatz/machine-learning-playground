# Tensorflow Playground

TensorFlow で各種学習モデルの動作確認を行うためのリポジトリです。

## 使ってみたモデル

- [TensorFlow Model Garden](https://github.com/tensorflow/models)
- [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)

## 必要なもの

- Python >=3.7
- Visual Studio Code

## インストール

```shell
$ git clone --recursive https://github.com/rinatz/tensorflow-models-docs-jp
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install -U pip
(.venv) $ pip install -r requirements.txt
```

VSCode で下記のどれかを開いてください。

- [object_detection.ipynb](object_detection.ipynb)
- [yolo.ipnynb](yolo.ipynb)
