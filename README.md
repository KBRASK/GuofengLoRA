# GuofengLoRA使用文档

本项目是一个中文项目，由于作者编写相关文档经验不足，如有格式上的指导或建议欢迎在 [Issues](https://github.com/KBRASK/GuofengLoRA/issues) 中指出。

## 目录
1. [简介](#简介)
3. [GuofengLoRA使用方法](#GuofengLoRA使用方法)
4. [LoRA模型训练方法](#LoRA模型训练方法)
5. [搭配SVD的使用方法](#搭配SVD的使用方法)
6. [搭配Anytext的使用方法](#搭配Anytext的使用方法)
7. [技术原理](#技术原理)
8. [开源工具致谢](#开源工具致谢)

# 简介
GuofengLoRA是一个基于SD1.5的拥有生成特定中国山水画画风的LoRA模型
其融合SVD模型后可以得到文生视频的能力
其融合Anytext模型后可以得到生成带有指定汉字的国风山水画

本项目提供其模型的两个版本及对应的两份带tag的训练集
说明利用的训练工具和推理使用的工具 并简述其训练思路及方法
并提供其与SVD模型或Anytext共同使用的方法。
# GuofengLoRA使用方法
## 简单文生图
利用简单的文字提示词生成对应的山水画
_本功能windows和linux可用_

### _实现该功能必备的工具有_
1.SD1.5基础模型
在下载lora模型前，确保你已经下载了[SD1.5的基础模型](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) 若只用于推理生图,下载emaonly版本即可,若用于训练LoRA模型,建议下载未修建版本的(7.7GB)

2.推理工具
推理工具建议使用B站up主[秋葉aaaki](https://space.bilibili.com/12566101)的[WebUI整合包](https://www.bilibili.com/video/BV1iM4y1y7oA/) 或者[ComfyUI](https://www.bilibili.com/video/BV1Ew411776J/)
前者适合新手使用,如果想使用GuofengLoRAwithSVD的文生视频功能,请使用ComfyUI
根据秋葉aaak的安装教程安装即可

3.GuofengLoRA模型
在本仓库的LoRA文件夹内获取,其中2.1版本生图稳定性更好,两者风格较有不同。

基础模型放在对应推理工具文件夹下的\\models\\Stable-diffusion内
GuofengLoRA模型放在\\models\\Lora文件夹下

### 1.WebUI下使用方法
![[example1.png]]
进入推理工具后,左上角的Stable Diffusion模型选择刚刚放入的基础模型

接下来在正向提示词部分选择扩展模型中一个版本的GuofengLoRA模型
在","后面写上生图像的正向提示词

推荐使用的正向提示词为: water,mountain
推荐使用的负面提示词为: trypophobia,
下面的steps建议设置为36-45
Sampler选择DPM++ 2M Karras
宽度和高度都选择512像素

### ComfyUI下使用方法
下载仓库中template中的texttoimageworkflow.json 导入即可

# LoRA模型训练方法
正在编写......
# 搭配SVD的使用方法
正在编写......
# 搭配Anytext的使用方法
正在编写......
# 技术原理
正在编写......
# 开源工具致谢
正在编写......
