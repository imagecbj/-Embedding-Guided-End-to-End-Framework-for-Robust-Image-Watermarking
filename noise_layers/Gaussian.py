import numpy as np
import math

from PIL.Image import Image

from PIL import Image as p
import numpy as np
from time import clock
import math
import torch.nn as nn
import torch
import random


# define
sizepic = [0, 0]
timer = [0, 0, 0, 0]
PI = 3.1415926

from PIL import Image
from matplotlib.pyplot import imsave
import numpy as np
from numpy import pi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from kornia.filters import GaussianBlur2d

from time import clock


def KernelMaker(r):  ### r 是半径
    '''高斯分布卷积核'''

    kernel = np.empty((2 * r + 1, 2 * r + 1))
    summat = 0  ## 矩阵求和，用来归一化
    for i in range(0, 2 * r + 1):
        for j in range(0, 2 * r + 1):
            dr2 = 2 * (r ** 2)  # double r^2
            gaussp = (1 / (pi * dr2)) * np.exp(-((i - r) ** 2 + (j - r) ** 2) / (dr2))
            kernel[i][j] = gaussp
            summat += gaussp

    kernel = kernel / summat  ## 归一化
    #print("高斯函数矩阵为\n", kernel)
    return kernel


def test_kernelCreate():
    r = 10
    kernel = KernelMaker(r)
    x_values = np.arange(-r, r + 1, 1)
    y_values = np.arange(-r, r + 1, 1)
    X, Y = np.meshgrid(x_values, y_values)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, kernel)
    plt.show()


def Channel_partial(img):
    width = img.size[0]
    height = img.size[1]
    nR = np.empty((width, height))
    nG = np.empty((width, height))
    nB = np.empty((width, height))

    # for i in range(0,width):
    #     for j in range(0,height):
    #         nR[i][j]=img.getpixel((i,j))[0]
    #         nG[i][j]=img.getpixel((i,j))[1]
    #         nB[i][j]=img.getpixel((i,j))[2]
    imgArray = np.array(img)

    nR = imgArray[:, :, 0]
    nG = imgArray[:, :, 1]
    nB = imgArray[:, :, 2]
    return nR, nG, nB


def Channel_Compound(nR, nG, nB):
    width = nR.shape[0]
    height = nR.shape[1]
    imgArray = np.empty((width, height, 3))
    imgArray[:, :, 0] = nR
    imgArray[:, :, 1] = nG
    imgArray[:, :, 2] = nB
    imgArray = np.asarray(imgArray, np.uint8)
    img = Image.fromarray(imgArray)
    return img


def GaussianBlur(array, kernel, mode):
    '''高斯模糊'''
    Timer = clock()
    width = array.shape[0]
    height = array.shape[1]
    if mode == 'fft':
        F_array = np.fft.fft2(array)
        F_kernel = np.fft.fft2(kernel, s=(width, height))
        new_F_array = F_array * F_kernel
        new_array = np.fft.ifft2(new_F_array)
        new_array = np.asarray(new_array, np.uint8)
        return new_array

    if mode == 'conv':
        new_array = np.empty(array.shape)
        r = (kernel.shape[0] - 1) // 2
        for i in range(r + 1, width - r):
            for j in range(r + 1, height - r):
                o = 0
                for x in range(i - r, i + r + 1):
                    p = 0
                    for y in range(j - r, j + r + 1):
                        new_array[i][j] += array[x][y] * kernel[o][p]
                        p += 1
                    o += 1
        new_array = np.asarray(new_array, np.uint8)
        return new_array


# if __name__ == "__main__":
#     r = 1  ## 卷积核半径
#     kernel = KernelMaker(r)
#     ## 测试一下生成卷积核
#     # test_kernelCreate()
#     path = r'E:\Xilinx失焦\Blur\1.jpg'
#     img = Image.open(path)
#     nR, nG, nB = Channel_partial(img)
#
#     FnR = GaussianBlur(nR, kernel, 'fft')
#     FnG = GaussianBlur(nG, kernel, 'fft')
#     FnB = GaussianBlur(nB, kernel, 'fft')
#
#     newimg = Channel_Compound(FnR, FnG, FnB)
#     newimg.show()
#     newimg.show()

# kornia
class Gaussian(nn.Module):
    def __init__(self,sigma=2,kernel = 3):
       super(Gaussian, self).__init__()
       self.gussian_filter = GaussianBlur2d((11,11),(sigma,sigma))



    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noise = self.gussian_filter(noised_image)
        noised_and_cover[0] = noise

        # batch, channel, col, row = noised_image.shape
        # for i in range(batch):
        #
        #     noise = self.gussian_filter(noised_image[i])
        #     noised_image[i] = noise
        # noised_and_cover[0] = noised_image

        return noised_and_cover



# class Gaussian(nn.Module):
#     def __init__(self,kernel_range):
#        super(Gaussian, self).__init__()
#        self.kernel_min = kernel_range[0]
#        self.kernel_max = kernel_range[1]
#
#
#     def forward(self, noised_and_cover):
#         noised_image = noised_and_cover[0]
#         batch,channel,col,row = noised_image.shape
#         #r = random.randint(1,10)
#         r = 1
#         kernel = KernelMaker(r)
#         for i in range(batch):
#             r = noised_image[i][0].detach().cpu().numpy()
#             g = noised_image[i][1].detach().cpu().numpy()
#             b = noised_image[i][2].detach().cpu().numpy()
#
#             FnR = GaussianBlur(r, kernel, 'fft')
#             FnG = GaussianBlur(g, kernel, 'fft')
#             FnB = GaussianBlur(b, kernel, 'fft')
#
#             FnRT = torch.from_numpy(FnR)
#             FnGT = torch.from_numpy(FnG)
#             FnBT = torch.from_numpy(FnB)
#
#             noised_image[i][0] = FnRT
#             noised_image[i][1] = FnGT
#             noised_image[i][2] = FnBT
#         #print(noised_image.shape)
#
#         #读RGB三通道分别做高斯模糊
#
#
#         return noised_and_cover


