# import numpy as np
# import torch
# import os
# import torch.nn as nn
# import pytorch_ssim
# from torch.autograd import Variable
# from torch import optim
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
# img2 = torch.rand(img1.size())
#
# img1 = Variable(torch.rand(1, 1, 256, 256))
# img2 = Variable(torch.rand(1, 1, 256, 256))
# print(img2.dtype)
# if torch.cuda.is_available():
#     img1 = img1.cuda()
#     img2 = img2.cuda()
#     print(1)
# #temp = pytorch_ssim.ssim(img1, img2,window_size = 11)
# #print(temp)
#     #pytorch_ssim.ssim(img1, img2)
#     #ssim_value = 1-pytorch_ssim.ssim(img1, img2).item()
#     ssim_loss = pytorch_ssim.SSIM().to(device)
#     #print(ssim_loss(img1, img2))
#
# optimizer = optim.Adam([img2], lr=0.01)
# ssim_value = 1
# while ssim_value > 0.05:
#     optimizer.zero_grad()
#     ssim_out = 1-ssim_loss(img1, img2).to(device)
#     #ssim_value = ssim_out.item()
#     print(ssim_value)
#     ssim_out.backward()
#     optimizer.step()
#     print(1)

import pytorch_ssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np

npImg1 = cv2.imread("einstein.png")

img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
img2 = torch.rand(img1.size())

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()


img1 = Variable( img1,  requires_grad=False)
img2 = Variable( img2, requires_grad = True)


# Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
ssim_value = pytorch_ssim.ssim(img1, img2).item()#data[0]
print("Initial ssim:", ssim_value)

# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
ssim_loss = pytorch_ssim.SSIM()

optimizer = optim.Adam([img2], lr=0.01)

while ssim_value < 0.95:
    optimizer.zero_grad()
    ssim_out = -ssim_loss(img1, img2)
    ssim_value = - ssim_out.item()#.data[0]
    print(ssim_value)
    ssim_out.backward()
    optimizer.step()