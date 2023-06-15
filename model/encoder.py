import torch
import torch.nn as nn
import sys
sys.path.append('../')
from options import HiDDenConfiguration
import cv2
import numpy as np


def conv_layer(chann_in, chann_out,k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding = p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer


class Encoder(nn.Module):
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        
        self.num_blocks = config.encoder_blocks



        #灰度message卷积层
        self.layer1 = conv_layer(1,  16, 3, 1)
        self.layer2 = conv_layer(16, 16, 3, 1)
        self.layer3 = conv_layer(16, 16, 3, 1)
        self.layer4 = conv_layer(16, 16, 3, 1)
        self.layer5 = conv_layer(16, 16, 3, 1)
        self.layer6 = conv_layer(16, 16, 3, 1)
        self.layer7 = conv_layer(16, 16, 3, 1)

        #彩色图像image卷积层
        self.layera = conv_layer(3, 16, 3, 1)
        self.layerb = conv_layer(32, 16, 3, 1)
        self.layerc = conv_layer(16, 32, 3, 1)
        self.layerd = conv_layer(48, 16, 3, 1)
        self.layere = conv_layer(16, 32, 3, 1)
        self.layerf = conv_layer(48, 16, 3, 1)
        self.layerg = conv_layer(16, 32, 3, 1)

        self.layer8 = conv_layer(48, 3, 1, 0)




    def forward(self, image, message):
        # expanded_message = message.unsqueeze(-1)
        # expanded_message.unsqueeze_(-1)
        #
        # message = expanded_message.expand(-1, -1, self.H, self.W)

        #第1层
        # print(image.shape)
        # print(message.shape)
        # for i in range(image.shape[0]):
        #     img = image[i].permute(1,2,0)
        #     cv2.imwrite("/new/zbb/temp/"+"img"+str(i)+".png", img.cpu().numpy())
        #     mess = message[i].permute(1,2,0)
        #     cv2.imwrite("/new/zbb/temp/"+"mess"+str(i)+".png", mess.cpu().numpy())

        # 拿到权重
        #print(message[0][0].detach().cpu().numpy())
        batch, channel, col, row = image.shape
        print('image:',image.shape)
        print('message:',message.shape)
        n_message = message.clone()
        n_image = image.clone()
        # #对每一个batch拿到一个权重 再根据这个权重对水印做排序
        for i in range(batch):
            #n_message[i][0] = 0
            image_np = n_image[i].detach().cpu().numpy()  #张量转图片
            image_np = np.transpose(image_np,(1,2,0))
            image_np = image_np*255.0  #载体图像0-255
            # if(i == 0):
            #     print(image_np[2])
            # print(image_np.shape)
            # print(image_np[0])

            # print(image_np)
            # print(image_np.shape)
            # print('####################')
            # print(i)
            image_Ycbcr = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCrCb)
            #Y,Cb,Cr = cv2.split(image_Ycbcr)
            #CbCr = Cb+Cr
            #print(np.min(Cb))
            #print(Cb)
            # 加色度

            B,G,R = cv2.split(image_np)
            Cb = -0.1687*R - 0.3313*G + 0.5*B + 128
            Cr = 0.5*R - 0.4187*G - 0.0813*B + 128
            CbCr = Cb + Cr
            Sc = ((Cb+Cr)-np.min(CbCr))/(np.max(CbCr)-np.min(CbCr)+1e-6)
            #print(Sc)


            gray = cv2.cvtColor(image_Ycbcr, cv2.COLOR_BGR2GRAY)  # 将图像转为灰度图像
            gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 将图像进行高斯模糊
            gray= gray.astype( np.uint8 )
            edge = cv2.Canny(gray, 1, 23)  # Canny边缘检测  0-255
            #print(edge)
            weight = edge / 255.0
            weight = (weight+Sc)/2
            #print(weight)

            #x, y = weight.shape
            # for m in range(x):
            #  for n in range(y):
            #          print(edge[m][n])
        #     #print(edge.shape)
        #
        #     # 拿出水印
            message_np = n_message[i][0].detach().cpu().numpy()
            message_np = message_np - (1 - weight)
            max_m = np.max(message_np)
            min_m = np.min(message_np)

            # print(max_m)
            # print(min_m)
            message_np = (message_np-min_m/(max_m - min_m+1e-6)) * max_m
            # print(message_np)
            # x, y = message_np.shape
            # for m in range(x):
            #     for n in range(y):
            #         if (message_np[m][n] < 0):
            #             message_np[m][n] = 0
            message_np = torch.from_numpy(message_np)
            n_message[i][0] = message_np

        #
            # 排序
            # edge_f = edge.flatten()
            # message_f = message_np.flatten()
            # message_s = np.sort(message_f)
            #
            # edge_index = np.argsort(edge_f)
            #
            # new_message = np.ones(message_s.shape)
            # j = 0
            # for x in edge_index:
            #     new_message[x] = message_s[j]
            #     j = j + 1
            #
            # new_message = new_message.reshape((128, 128))
            # message_np = new_message
            # message_np = torch.from_numpy(message_np)
            # n_message[i][0] = message_np

        # print(message.shape)
        # print(type(message))
        # print(type(image))
        # print(image.shape)

        #     noised_image_np = gaussnoise(noised_image_np,scale)#图片做高斯
        #     noised_image_np = torch.from_numpy(noised_image_np)#图片转张量
        #     noised_image[i] = noised_image_np
        # noised_and_cover[0] = noised_image


        out1_1 = self.layer1(n_message)
        out2_2 = self.layera(image)
        out = torch.cat((out1_1, out2_2), 1)

        # 第2层
        out1_2 = self.layer2(out1_1)
        out2_2 = self.layerb(out)

        # 第3层
        out1_3 = self.layer3(out1_2)
        out2_3 = self.layerc(out2_2)
        out = torch.cat((out1_3, out2_3), 1)

        # 第4层
        out1_4 = self.layer4(out1_3)
        out2_4 = self.layerd(out)

        # 第5层
        out1_5 = self.layer5(out1_4)
        out2_5 = self.layere(out2_4)
        out = torch.cat((out1_5, out2_5), 1)

        # 第6层
        out1_6 = self.layer6(out1_5)
        out2_6 = self.layerf(out)

        # 第7层
        out1_7 = self.layer7(out1_6)
        out2_7 = self.layerg(out2_6)
        out = torch.cat((out1_7, out2_7), 1)

        out = self.layer8(out)


        return out































