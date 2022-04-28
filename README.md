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
DBNet针对基于分割的方法需要使用阈值进行二值化处理而导致后处理耗时的问题，提出了可学习阈值并巧妙地设计了一个近似于阶跃函数的二值化函数，使得分割网络在训练的时候能端对端的学习文本分割的阈值。自动调节阈值不仅带来精度的提升，同时简化了后处理，提高了文本检测的性能。
![](https://ai-studio-static-online.cdn.bcebos.com/0d6423e3c79448f8b09090cf2dcf9d0c7baa0f6856c645808502678ae88d2917)

Jupyter Notebook编译环境，详情查看[Detection](./det_mv3_db/Detection.md)

# 二、中文文本识别


# 结语
