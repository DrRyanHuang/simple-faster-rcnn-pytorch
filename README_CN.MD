[English](https://github.com/DrRyanHuang/simple-faster-rcnn-pytorch/blob/master/README.MD)|中文



# 你之前木有见过的简单快速的Faster R-CNN实现

## 1. 引言

**[更新:]** 笔者已经用`pytorch 1.5 torchvision 0.6`简化了代码，用`torchvision`的函数替代了源代码中的`roipool`和`nms`操作. 如果老铁想阅读旧版本的代码，可以来看分支 [v1.0](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/tree/v1.0)

本项目基于[chainercv](https://github.com/chainer/chainercv) 和其他[项目](#acknowledgement)实现了一个简化版的`Faster R-CNN`. 笔者希望本代码可以对那些想对`Faster R-CNN`有所了解的小白提供帮组. 本代码旨在于：

- 简化代码（*简单胜于复杂*）
- 使代码更直接（*顺序结构 [这里不太会翻译，有老铁看见帮我组织下语言] 优于嵌套*）
- 与[原论文](https://arxiv.org/abs/1506.01497)结果相匹配（*速度和`mAP`*）



同时，本代码也有以下优点:
- 本项目可以直接以Python代码执行，不再需要其他操作[这里不太会翻译，有老铁看见帮我组织下语言]
- 仅仅约2000行代码就实现了`Faster RCNN`，并且带有大量注释和说明（这多亏了`chainercv`出色的文档）
- `mAP`上取得了更好的结果（比原论文结果还nice，0.712 VS 0.699）
- 他的运行速度和其他实现相当（在`TITAN XP`上，训练为`6fps`，测试为`14fps`）
- 能够有效利用内存（使用`vgg16`大约只用3G）


![img](imgs/faster-speed.jpg)



## 2. 运行表现

### 2.1 `mAP`上

`VGG16`在`VOC2007 trainval`上训练，而在`VOC2007 test`上测试.

**注意**: 训练显然有很大的随机性，可能需要一点运气和更多的训练时间才能达到最高的`mAP`，但是应该很容易超过下限嘿嘿

|                        Implementation                        |   `mAP`值   |
| :----------------------------------------------------------: | :---------: |
|          [原论文](https://arxiv.org/abs/1506.01497)          |    0.699    |
|                使用`caffe`预训练模型进行训练                 | 0.700-0.712 |
|             使用`torchvision`预训练模型进行训练              | 0.685-0.701 |
| 使用由[chainercv](https://github.com/chainer/chainercv/tree/master/examples/faster_rcnn)导出的模型 (原项目中mAP为0.706) |   0.7053    |

### 2.2 速度上

|                        Implementation                        |   `GPU`    | Inference | Trainining |
| :----------------------------------------------------------: | :--------: | :-------: | :--------: |
|       [origin paper](https://arxiv.org/abs/1506.01497)       |   `K40`    |   5 fps   |     NA     |
|                           This[1]                            | `TITAN Xp` | 14-15 fps |   6 fps    |
| [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) | `TITAN Xp` | 15-17 fps |   6 fps    |

[1]:  确保正确安装`Cupy`，并且`GPU`上仅运行这一个程序.  训练速度对您的`GPU`状态很敏感。 

有关更多信息，请参见[问题解决](问题解决).  此外，在程序启动时它很——需要时间进行预热。

通过关闭可视化和写日志等功能可以使程序运行更快.



## 3. 安装依赖环境

这是一个使用`anaconda`从头开始创建环境的示例：

```sh
# create conda env
conda create --name simp python=3.7
conda activate simp
# install pytorch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# install other dependancy
pip install visdom scikit-image tqdm fire ipdb pprint matplotlib torchnet

# start visdom
nohup python -m visdom.server &
```

如果老铁你不用`anaconda`，那么：

- 安装`GPU`版本的`Pytorch`(本代码只用在GPU上), refer to [official website](http://pytorch.org)
- 安装其他依赖包:  `pip install visdom scikit-image tqdm fire ipdb pprint matplotlib torchnet`
- 开启`visdom`服务以进行可视化

```
nohup python -m visdom.server &
```



## 4. Demo

可以从 [Google Drive](https://drive.google.com/open?id=1cQ27LIn-Rig4-Uayzy_gH5-cW-NRGVzY) 或者 [Baidu Netdisk(passwd: scxn)](https://pan.baidu.com/s/1o87RuXW) 下载预训练模型

可以查看 [demo.ipynb](https://github.com/DrRyanHuang/simple-faster-rcnn-pytorch/blob/master/demo.ipynb) 以获取更多细节.



## 5. 训练

### 5.1 数据准备

#### `Pascal VOC2007`

1. 下载`VOC training, validation, test`数据集，别忘了 `VOCdevkit`

   ```Bash
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
   ```

2. 解压所有这些tar文件，并将其放在`VOCdevkit`文件夹中

   ```Bash
   tar xvf VOCtrainval_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar
   tar xvf VOCdevkit_08-Jun-2007.tar
   ```

3. 文件的结构应该是这样的：

   ```Bash
   $VOCdevkit/                           # development kit
   $VOCdevkit/VOCcode/                   # VOC utility code
   $VOCdevkit/VOC2007                    # image sets, annotations, etc.
   # ... and several other directories ...
   ```

4. 修改`utils/config.py`文件中的 `voc_data_dir` 配置项，或者运行程序通过`--voc-data-dir=/path/to/VOCdevkit/VOC2007/` 来传入参数




### 5.2 [可选]准备`caffe`预训练vgg16模型

如果要使用经过`caffe`训练的模型作为初始权重，则可以运行下面的代码修改`vgg16`的权重参数，没错，和原论文是一样滴

````Bash
python misc/convert_caffe_pretrain.py
````

该脚本将下载预训练模型并将其转换为与`Torchvision`兼容的格式，如果你在中国且不能下载这个预训练模型，那么可以参考这个[issue](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/issues/63)

然后，您可以通过设置`utils/config.py`中的`caffe_pretrain_path`参数来指定`caffe`预训练模型`vgg16_caffe.pth`的存储位置. 默认路径为也是hin可以的.

如果你想要用`torchvision`的预训练模型，则可以跳过此步骤.

**注1**：`caffe`预训练模型已表现出更好的性能。

**注2**：`caffe`模型要求图片是BGR 0-255的，而`torchvision`模型要求图片是RGB 0-1的. 具体可以查看`data/dataset.py`



### 5.3 开始训练


```bash
python train.py train --env='fasterrcnn' --plot-every=100
```

你可以参看 `utils/config.py` 以获取更多参数

这里是一些关键参数:

- `--caffe-pretrain=False`: 使用来自 caffe 还是 torchvision的预训练参数 (默认: torchvison)
- `--plot-every=n`: 每`n`个batch，可视化一次预测结果与loss值等
- `--env`: 用于可视化的 `visdom`环境
- `--voc_data_dir`:  VOC 数据保存的位置
- `--use-drop`: 是否在RoI头中使用`dropout`，默认是`False`
- `--use-Adam`: 使用`Adam`而不是`SGD`, 默认是`SGD`. (要是使用Adam,则`lr`应设置得很小)
- `--load-path`: 预训练模型路径，默认为`None`，如果被指定，那么模型将被加载

你可以打开浏览器, 输入`http://<ip>:8097`，就能看到如下所示的训练过程可视化图片：

![visdom](imgs/visdom-fasterrcnn.png)



## 问题解决

- dataloader: `received 0 items of ancdata` 

  可以查看这个[discussion](https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667)，不过他已经被在 [train.py](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/train.py#L17-L22) 中修改了，所以我觉得你可能遇不到这个问题了
  
- 关于 Windows上支持

  由于笔者没有有`GPU`的windows系统去debug和测试这份代码. 所以如果有人可以帮助我在windows上测试和PR，那将是十分受欢迎的！！



## Acknowledgement
这个项目建立在很多十分nice的项目之上，包括：

- [Yusuke Niitani's ChainerCV](https://github.com/chainer/chainercv) (主要)
- 基于[Xinlei Chen's tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) 的 [Ruotian Luo's pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) 
- [faster-rcnn.pytorch by Jianwei Yang and Jiasen Lu](https://github.com/jwyang/faster-rcnn.pytorch). 该项目主要参考自 [longcw's faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch)
- 所有以上的项目都直接或间接参考了项目 [py-faster-rcnn by Ross Girshick and Sean Bell](https://github.com/rbgirshick/py-faster-rcnn)



## ^_^
本项目为MIT协议，查看LICENSE以获取更多细节！

十分欢迎 Contribution !

如果老哥遇到什么任何问题，可随时开一个issue，但是最近有些忙可能回复晚一些嘿嘿

如果我有哪里错了或者说的不清楚，还请不吝赐教！

介是模型架构图：
![img](imgs/model_all.png)