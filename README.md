# PaddleOCR-for-Chinese-Text

# 前言
OCR（Optical Character Recognition）即光学字符识别的缩写，可应用于文档资料、证件票据等场景，针对复杂图文场景通常把任务分解为（文本检测+文本识别）两个子任务。中文是承载中国文化的重要工具，携带着丰富且精确的信息，在日常生活中经常能见到带有中文的图像或视频，挖掘其中的信息能提升人民数字化生活的便捷性。

- 中文文本检测（Detection）是找出图像或视频中文字的位置
![](https://ai-studio-static-online.cdn.bcebos.com/400b9100573b4286b40b0a668358bcab9627f169ab934133a1280361505ddd33)

- 中文文本识别（Recognition）是将图像信息转换为文字信息
![](https://ai-studio-static-online.cdn.bcebos.com/a7c3404f778b489db9c1f686c7d2ff4d63b67c429b454f98b91ade7b89f8e903)

难点在于：

* 自然场景中文本具有多样性：文本检测受到文字颜色、大小、字体、形状、方向、语言、以及文本长度的影响；

* 复杂的背景和干扰；文本检测受到图像失真，模糊，低分辨率，阴影，亮度等因素的影响；

* 文本密集甚至重叠会影响文字的检测；

* 文字存在局部一致性，文本行的一小部分，也可视为是独立的文本；

# 一、中文文本检测
## 1.1 安装PaddleOCR
```
#下载PaddleOCR
!git clone https://gitee.com/paddlepaddle/PaddleOCR


#安装配置文件
%cd PaddleOCR
!pip install --upgrade pip
!pip install -r "requirements.txt" -i https://mirror.baidu.com/pypi/simple
```

## 1.2 数据集准备
det_data_lesson_demo数据集是中文文本检测通用数据集，包含3009张训练图像和3009张测试图像，场景分别为街景图片、商用广告和电子文档，数据集目录如下所示：
```
|---lsvt
    |---train
    |---eval
|---mtwi
    |---train
    |---eval
|---xfun
    |---train
    |---val
    |---zh_det_train.txt
    |---zh_det_val.txt
eval.txt
train.txt
```
数据集下载链接：
[https://paddleocr.bj.bcebos.com/dataset/det_data_lesson_demo.tar](https://paddleocr.bj.bcebos.com/dataset/det_data_lesson_demo.tar)

- lsvt-街景图片

![](https://ai-studio-static-online.cdn.bcebos.com/e8e804b90957441ca32fc54787358cdc170592e003b3490b9ebb92e486f5f218)

- mtwi-商用广告


![](https://ai-studio-static-online.cdn.bcebos.com/8e7e451ec91a453cae8873a356bbebc8854fcfa1838a4de091c5e26400aa5fb3)

- xfun-电子文档

![](https://ai-studio-static-online.cdn.bcebos.com/ad4944ff47b04b85898fcb8651bf26358b86a2e786bf45f88a2eea2b02c03335)

## 1.3 预训练模型

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|ch_PP-OCRv2_det_slim|【最新】slim量化+蒸馏版超轻量模型，支持中英文、多语种文本检测|[ch_PP-OCRv2_det_cml.yml](../../configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_cml.yml)| 3M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_quant_infer.tar)|
|ch_PP-OCRv2_det|【最新】原始超轻量模型，支持中英文、多语种文本检测|[ch_PP-OCRv2_det_cml.yml](../../configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_cml.yml)|3M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar)|
|ch_ppocr_mobile_slim_v2.0_det|slim裁剪版超轻量模型，支持中英文、多语种文本检测|[ch_det_mv3_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml)| 2.6M |[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_prune_infer.tar)|
|ch_ppocr_mobile_v2.0_det|原始超轻量模型，支持中英文、多语种文本检测|[ch_det_mv3_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml)|3M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar)|
|ch_ppocr_server_v2.0_det|通用模型，支持中英文、多语种文本检测，比超轻量模型更大，但效果更好|[ch_det_res18_db_v2.0.yml](../../configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml)|47M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar)|

```
#下载预训练模型ch_ppocr_server_v2.0_det_train
!wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar
!tar -xvf ch_ppocr_server_v2.0_det_train.tar
!mkdir pretrain_models
!mv ch_ppocr_server_v2.0_det_train ./pretrain_models/ch_ppocr_server_v2.0_det_train
!rm ch_ppocr_server_v2.0_det_train.tar
```

## 1.4 启动训练
PaddleOCR有两种配置文件方式

`-c`指定配置文件路径

`-o`修改配置文件参数

配置文件参数介绍查询[链接](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_ch/config.md)

如果训练程序中断，如果希望加载训练中断的模型从而恢复训练，可以通过指定`Global.checkpoints`指定要加载的模型路径。需要注意`Global.checkpoints`的优先级高于`Global.pretrained_model`的优先级，即同时指定两个参数时，优先加载`Global.checkpoints`指定的模型，如果`Global.checkpoints`指定的模型路径有误，会加载`Global.pretrained_model`指定的模型。

```
#DB
!python tools/train.py -c configs/det/det_r50_vd_db.yml \
    -o Global.use_visualdl=True \
    Global.epoch_num=100    \
    Global.save_epoch_step=10   \
    Global.eval_batch_step=[0,200]   \
    Global.print_batch_step=20  \
    Global.pretrained_model='./pretrain_models/ch_ppocr_server_v2.0_det_train/best_accuracy.pdparams'  \
    Train.dataset.data_dir='./train_data/det_data_lesson_demo/' \
    Train.dataset.label_file_list=['./train_data/det_data_lesson_demo/train.txt']   \
    Train.loader.batch_size_per_card=16 \
    Train.loader.num_workers=0  \
    Eval.dataset.data_dir='./train_data/det_data_lesson_demo/'  \
    Eval.dataset.label_file_list=['./train_data/det_data_lesson_demo/eval.txt'] \
    Eval.loader.batch_size_per_card=1   \
    Eval.loader.num_workers=0   \
```

## 1.5 模型评估
PaddleOCR计算三个OCR检测相关的指标，分别是：Precision、Recall、Hmean（F-Score）。

训练中模型参数默认保存在`Global.save_model_dir`目录下。在评估指标时，需要设置`Global.checkpoints`指向保存的参数文件。

```
#DB
!python tools/eval.py -c output/det_r50_vd/config.yml -o Global.checkpoints='output/det_r50_vd/best_accuracy.pdparams'
```

## 1.6 模型预测
预测单张图片

```
python3 tools/infer_det.py -c configs/det/det_mv3_db.yml \
    -o Global.infer_img="./doc/imgs_en/img_10.jpg"\
    Global.checkpoints="./output/det_db/best_accuracy"
```

预测文件夹下所有图片

```
python3 tools/infer_det.py -c configs/det/det_mv3_db.yml \
    -o Global.infer_img="./doc/imgs_en/"\
    Global.checkpoints="./output/det_db/best_accuracy"
```

预测结果图片保存在路径'PaddleOCR/output/det_db/det_results/'下：

PaddleOCR/output/det_db/det_results/lsvt.jpg

PaddleOCR/output/det_db/det_results/mtwi.jpg

PaddleOCR/output/det_db/det_results/xfun.jpg

## 1.7 模型推理
inference 模型（paddle.jit.save保存的模型） 一般是模型训练，把模型结构和模型参数保存在文件中的固化模型，多用于**预测部署场景**。 训练过程中保存的模型是checkpoints模型，保存的只有模型的参数，多用于恢复训练等。 与checkpoints模型相比，inference 模型会额外保存模型的结构信息，在预测部署、加速推理上性能优越，灵活方便，适合于实际系统集成。
```
#DB
!python tools/export_model.py -c output/det_r50_vd/config.yml \
    -o Global.checkpoints='output/det_r50_vd/best_accuracy.pdparams'   \
    Global.save_inference_dir='output/det_r50_vd/inference/'

!python tools/eval.py -c output/det_r50_vd/config.yml \
    -o Global.checkpoints='output/det_r50_vd/inference/inference.pdiparams'
```

## 1.8 结果保存



# 二、中文文本识别


# 结语
