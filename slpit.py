
import os
import torch
from torch import nn
from model.hidden import Hidden
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser
from train import train
from torchvision.models import alexnet

import numpy as np
import torch
import torch.nn as nn
import pytorch_ssim
import sys,os
import torchsnooper
import cv2
import utils
from tensorboard_logger import TensorBoardLogger
import torch
import torch.nn
import argparse
import os
import numpy as np
from options import HiDDenConfiguration

import utils
from model.hidden import *
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF
import kornia

import SSIM

import MakeDataset

import logging

#os.environ['CUDA_VISIBLE_DEVICES'] ='1'






# 核心函数，参考了torch.quantization.fuse_modules()的实现
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

# 以AlexNet为例子


# model = hidden_net()
#
# # 打印原模型
# print("原模型")
# print(model)


# 打印每个层的名字，和当前配置
# 从而知道要改的层的名字
# for name in model.state_dict():
#     print(name)
#
# # 假设要换掉AlexNet前2个卷积层，将通道从64改成128，其余参数不变
# # 定义新层
# layer0 = nn.Conv2d(3, 128, (11, 11), (4, 4), (2,2))
# layer1 = nn.Conv2d(128, 192, (5, 5), (1, 1), (2,2))
# # 层的名字从上面19-20行的打印内容知道AlexNet前2个层的名字为 "features.0" 和 "features.3"
# _set_module(model, 'features.0', layer0)
# _set_module(model, 'features.3', layer1)
#
# # 打印修改后的模型
# print("新模型")
# print(model)
#
# # 推理试一下
# img = torch.rand((1, 3, 224, 224))
# model(img)


if __name__ == '__main__':
    main()
