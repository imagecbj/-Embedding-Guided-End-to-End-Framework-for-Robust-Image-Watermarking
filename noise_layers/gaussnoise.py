import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import random


def gaussnoise(image,scale):

    #image = cv2.imread(filename)
    noise = np.random.normal(loc=0.0, scale=scale, size=(128, 128))  # mean 0.0, std 5
    decimg = np.clip(np.asarray(image, float) + np.tile(np.expand_dims(noise, 0), (3, 1, 1)), 0.0, 255.0)

    return decimg

class Gaussnoise(nn.Module):

    def __init__(self,ratio_range):
        super(Gaussnoise, self).__init__()

        self.ratio_min = ratio_range[0]
        self.ratio_max = ratio_range[1]

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        batch, channel, col, row = noised_image.shape

        #scale = random.randint(1, 3)
        scale = 1

        for i in range(batch):

            #r = 5
            noised_image_np = noised_image[i].detach().cpu().numpy()  #张量转图片
            # print(noised_image_np.shape)
            noised_image_np = gaussnoise(noised_image_np,scale)#图片做高斯
            noised_image_np = torch.from_numpy(noised_image_np)#图片转张量
            noised_image[i] = noised_image_np
        noised_and_cover[0] = noised_image

        ## 不要硬编码

        return noised_and_cover

