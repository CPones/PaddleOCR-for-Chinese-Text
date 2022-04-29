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
rec_data_lesson_demo数据集是中文文本检测通用数据集，属于SimpleDataSet类型，包含10w张训练图像和3000张测试图像，场景分别为街景图片、商用广告和电子文档，数据集目录如下所示：
```
|---lsvt
    |---lsvt_train_images
    |---lsvt_val_images
|---mtwi
    |---mtwi_train_images
    |---mtwi_val_images
|---xfun
    |---xfun_train
    |---xfun_val
val.txt
train.txt
```
数据集下载链接：
[https://paddleocr.bj.bcebos.com/dataset/rec_data_lesson_demo.tar](https://paddleocr.bj.bcebos.com/dataset/rec_data_lesson_demo.tar)

```
#下载数据集det_data_lesson_demo
!wget https://paddleocr.bj.bcebos.com/dataset/rec_data_lesson_demo.tar
!tar -xvf rec_data_lesson_demo.tar
!mkdir train_data
!mv rec_data_lesson_demo ./train_data/
!rm rec_data_lesson_demo.tar
```

## 1.3 预训练模型
|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|ch_PP-OCRv2_rec_slim|【最新】slim量化版超轻量模型，支持中英文、数字识别|[ch_PP-OCRv2_rec.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec.yml)| 9M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_train.tar) |
|ch_PP-OCRv2_rec|【最新】原始超轻量模型，支持中英文、数字识别|[ch_PP-OCRv2_rec.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec.yml)|8.5M|[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar) |
|ch_ppocr_mobile_slim_v2.0_rec|slim裁剪量化版超轻量模型，支持中英文、数字识别|[rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml)| 6M |[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_train.tar) |
|ch_ppocr_mobile_v2.0_rec|原始超轻量模型，支持中英文、数字识别|[rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml)|5.2M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar) |
|ch_ppocr_server_v2.0_rec|通用模型，支持中英文、数字识别|[rec_chinese_common_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml)|94.8M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_pre.tar) |

**说明：** `训练模型`是基于预训练模型在真实数据与竖排合成文本数据上finetune得到的模型，在真实应用场景中有着更好的表现，`预训练模型`则是直接基于全量真实数据与合成数据训练得到，更适合用于在自己的数据集上finetune。

```
#下载预训练模型ch_ppocr_server_v2.0_rec_train
!wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar
!tar -xvf ch_ppocr_server_v2.0_rec_train.tar
!mkdir pretrain_models
!mv ch_ppocr_server_v2.0_rec_train ./pretrain_models/ch_ppocr_server_v2.0_rec_train
!rm ch_ppocr_server_v2.0_rec_train.tar
```

## 1.4 启动训练
PaddleOCR有两种配置文件方式

`-c`指定配置文件路径

`-o`修改配置文件参数

配置文件参数介绍查询[链接](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_ch/config.md)

如果训练程序中断，如果希望加载训练中断的模型从而恢复训练，可以通过指定`Global.checkpoints`指定要加载的模型路径。需要注意`Global.checkpoints`的优先级高于`Global.pretrained_model`的优先级，即同时指定两个参数时，优先加载`Global.checkpoints`指定的模型，如果`Global.checkpoints`指定的模型路径有误，会加载`Global.pretrained_model`指定的模型。

```
#基于CTC(RCNNN)
!python tools/train.py -c configs/rec/rec_r34_vd_none_bilstm_ctc.yml    \
    -o Global.use_visualdl=True \
    Global.epoch_num=500    \
    Global.save_epoch_step=10   \
    Global.eval_batch_step=[0,200]   \
    Global.print_batch_step=20  \
    Global.character_dict_path='./ppocr/utils/ppocr_keys_v1.txt'    \
    Global.pretrained_model='./pretrain_models/ch_ppocr_server_v2.0_rec_train/best_accuracy.pdparams'  \
    Train.dataset.name=SimpleDataSet    \
    Train.dataset.data_dir='./train_data/rec_data_lesson_demo' \
    Train.dataset.label_file_list=['./train_data/rec_data_lesson_demo/train.txt']   \
    Train.loader.batch_size_per_card=256 \
    Train.loader.num_workers=0  \
    Eval.dataset.name=SimpleDataSet \
    Eval.dataset.data_dir='./train_data/rec_data_lesson_demo'  \
    Eval.dataset.label_file_list=['./train_data/rec_data_lesson_demo/val.txt'] \
    Eval.loader.batch_size_per_card=256   \
    Eval.loader.num_workers=0   \
```

## 1.5 模型评估
文本识别是一个分类任务，评估的指标是准确率(Accuracy)。在模型训练过程中需要指定一个字典，即以utf-8编码保存的字符类别，PaddleOCR内置了一部分字典，其中中文字典包含在目录`ppocr/utils/ppocr_key1_v1.txt`。

```
#基于CTC(CRNN)
!python tools/eval.py -c ./output/rec/r34_vd_none_bilstm_ctc/config.yml \
    -o Global.checkpoints='./output/rec/r34_vd_none_bilstm_ctc/best_accuracy.pdparams'
```

## 1.6 模型预测
默认预测图片存储在 `infer_img` 里，通过 `-o Global.checkpoints` 加载训练好的参数文件：

根据配置文件中设置的的 `save_model_dir` 和 `save_epoch_step` 字段，会有以下几种参数被保存下来：

```
output/rec/
├── best_accuracy.pdopt  
├── best_accuracy.pdparams  
├── best_accuracy.states  
├── config.yml  
├── iter_epoch_3.pdopt  
├── iter_epoch_3.pdparams  
├── iter_epoch_3.states  
├── latest.pdopt  
├── latest.pdparams  
├── latest.states  
└── train.log
```
其中 best_accuracy.* 是评估集上的最优模型；iter_epoch_x.* 是以 `save_epoch_step` 为间隔保存下来的模型；latest.* 是最后一个epoch的模型。

```
!python tools/infer_rec.py -c config.yml    \
    -o Global.infer_img='./doc/imgs_words/ch/word_1.jpg'    \
    Global.pretrained_model='best_accuracy.pdparams'
```

## 1.7 模型推理
* Global.pretrained_model 参数设置待转换的训练模型地址，不用添加文件后缀 .pdmodel，.pdopt或.pdparams。

* Global.save_inference_dir参数设置转换的模型将保存的地址。

注意：如果您是在自己的数据集上训练的模型，并且调整了中文字符的字典文件，请注意修改配置文件中的`character_dict_path`是否是所需要的字典文件。

转换成功后，在目录下有三个文件：

```
/inference/rec_crnn/
    ├── inference.pdiparams         # 识别inference模型的参数文件
    ├── inference.pdiparams.info    # 识别inference模型的参数信息，可忽略
    └── inference.pdmodel           # 识别inference模型的program文件
```

