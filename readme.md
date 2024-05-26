# YOLOV8(脱离ultralytics库)

本项目YOLOV8和官方相比，不需要安装ultralytics即可实现v8目标检测。训练和检测部分模仿yolov5进行的实现。

# 代码准备

git clone https://github.com/YINYIPENG-EN/YOLOV8.git

clone代码到本地

# 环境说明

torch==1.10.0+cu102

torchvision==0.11.0+cu102

numpy==1.22.4

# 训练自己的数据集

## 数据集准备

我这里的数据集是只有一个类，数据集名称叫target(根据自己的数据集自己命名)，然后我把我的数据集放在了cfg/datasets文件下。目录形式如下：

其中**Annotations存储的是xml形式的标签文件**，**images存储的所有的图像**，**labels存储的是将xml转为txt的标签文件**(也是我们需要用的)。

```
$ tree
|-- Annotations
|-- images
|-- labels
```

然后我们需要将images和labels划分成训练集和验证集。

修改split_dataset.py中的datasets_path路径，运行划分数据集脚本代码：

```
python split_dataset.py
```

划分完成后会在cfg/datasets/your_Datasets/下生成train和val两个文件夹，同时各包含images和labels子文件。

## 新建yaml文件

在cfg/datasets/下新建一个mydata.yaml文件，由于我这里只有一个类，而且类的name为"target"，因此配置文件内容如下：(这里**建议填写绝对路径**，否则可能会出现问题)

```
path: F:/YOLOV8/cfg/datasets/target
train: F:/YOLOV8/cfg/datasets/target/train/images
val: F:/YOLOV8/cfg/datasets/target/val/images
test: #
 
# number of classes
nc: 1
 
# class names
names:
  0: target
```

## 训练

训练代码在train.py中，这里传入的参数是模仿yolov5的同时并结合yolov8需要的参数实现的。

快速开启训练：

```shell
python train.py --weights yolov8s.pt --epochs 100 --bs 64 
```

也可以与其他训练参数搭配使用，这里介绍几个常用的参数：

--weights:预权重路径

--model:加载的yolov8类型，默认为yolov8s.yaml

--epochs:训练的epochs数量，默认100

--device:cuda训练

--cache:开启缓存，默认是开启的

--bs:batch size大小

--optimizer：优化器类型

--resume:继续训练

--freeze:冻结训练

# 检测

检测代码在detect.py中，快速开启检测

```shell
python detect.py --weights yolov8s.pt --source assets --show --save
```

也可以与其他参数搭配使用，介绍几种常用参数：

--weights:权重路径

--source:source路径，可以是图像、视频、文件夹

--visualize：特征可视化

--classes:仅检测特定的类别

--show:显示检测结果

--save:保存检测结果

--save_frames:如果是视频检测，开启后可以把每帧进行保存

--save_crop:开启后，可以把目标从背景中截出来并保存

