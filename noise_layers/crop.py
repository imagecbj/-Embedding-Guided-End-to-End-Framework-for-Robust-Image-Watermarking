# from random import random
#
# import torch.nn as nn
# import numpy as np
# import cv2
# import torch
#
#
# def random_attack(min, max):
#     """
#     Return a random number
#     :param min:
#     :param max:
#     :return:
#     """
#     return random.randint(min,max)
#
#
#
#
# class Crop(nn.Module):
#     """
#     Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
#     heigth_ratio_range and width_ratio_range
#     """
#     def __init__(self, height_ratio_range, width_ratio_range):
#         """
#
#         :param height_ratio_range:
#         :param width_ratio_range:
#         """
#         super(Crop, self).__init__()
#         self.height_ratio_min = height_ratio_range[0]
#         self.height_ratio_max = height_ratio_range[1]
#
#         self.width_ratio_min = width_ratio_range[0]
#         self.width_ratio_max = width_ratio_range[2]
#
#
#     def forward(self, noised_and_cover):
#         noised_image = noised_and_cover[0]
#         # crop_rectangle is in form (from, to) where @from and @to are 2D points -- (height, width)
#         # noised_image.shape
#         mask = np.ones((128, 128))
#         print(mask)
#         rows = random_attack(self.height_ratio_min, self.height_ratio_max)
#         columns = random_attack(self.width_ratio_min, self.width_ratio_max)
#
#         # 随机选择几行几列的交叉值改成0
#
#         mask_tensor = torch.tensor(mask, device=noised_image.device, dtype=torch.float)
#         mask_tensor = mask_tensor.expand_as(noised_image)
#         noised_image = noised_image * mask_tensor
#
#
#         noised_and_cover[0] = noised_image
#
#
#         ## 不要硬编码
#
#         return noised_and_cover
import random

import torch.nn as nn
import numpy as np

import torch


#获取随机裁剪的几行和几列
def randomRoworCol(x):
    begin = random.randint(1, 127)
    if(begin + x > 127):
        y = 127 -begin
        end = begin + y
        begin = begin - (x - y)
    else:
        end = begin + x
    return begin,end


#mask
def wudi(a,b):
    mask = np.ones(shape = (128,128))
    rowBegin, rowEnd = randomRoworCol(a)
    colBegin, colEnd = randomRoworCol(b)
    # print("a,b: ",a,b)
    # print(colBegin,colEnd)
    # print(rowBegin,rowEnd)
    for row in range(rowBegin, rowEnd):
        for col in range(colBegin, colEnd):
            mask[row][col] = 0
    return mask



class Crop(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
    heigth_ratio_range and width_ratio_range
    """
    def __init__(self, ratio_range):
        """

        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Crop, self).__init__()
        self.ratio_min = ratio_range[0]
        self.ratio_max = ratio_range[1]

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        # print(noised_image.shape)
        # for i in range(8):
        #     print(noised_image[i][0].shape)
        #     print(noised_image[i][1].shape)
        #     print(noised_image[i][2].shape)
        #     print(type(noised_image[i][0]))

        a = random.randint(self.ratio_min, self.ratio_max)
        b = random.randint(self.ratio_min, self.ratio_max)
        mask = wudi(50, 50)

        mask_tensor = torch.tensor(mask, device=noised_image.device, dtype=torch.float)
        mask_tensor = mask_tensor.expand_as(noised_image)
        noised_image = noised_image * mask_tensor
        noised_and_cover[0] = noised_image

        ## 不要硬编码

        return noised_and_cover