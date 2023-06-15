import torch.nn as nn
import sys
sys.path.append('../')
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
import torchvision
import torch


def conv_layer(chann_in, chann_out,k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding = p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer


class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, config: HiDDenConfiguration):

        super(Decoder, self).__init__()


        self.layer1 = conv_layer(3, 16, 3, 1)
        self.layer2 = conv_layer(16, 32, 3, 1)
        self.layer3 = conv_layer(32, 64, 3, 1)
        self.layer4 = conv_layer(64, 128, 3, 1)
        # self.layer5 = conv_layer(128, 256, 3, 1)
        # self.layer6 = conv_layer(256, 512, 3, 1)
        # self.layer7 = conv_layer(512, 256, 3, 1)
        # self.layer8 = conv_layer(256, 128, 3, 1)
        self.layer5 = conv_layer(128, 64, 3, 1)
        self.layer6 = conv_layer(64, 32, 3, 1)
        # self.layer11 = conv_layer(32, 3, 3, 1)
        # self.layer12 = conv_layer(3, 1, 3, 1)
        self.layer7 = conv_layer(32, 1, 3, 1)

    def forward(self, image_with_wm):
        out = self.layer1(image_with_wm)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.layer5(out)
        # out = self.layer6(out)
        # out = self.layer7(out)
        # out = self.layer8(out)
        out = self.layer5(out)
        out = self.layer6(out)
        # out = self.layer11(out)
        # out = self.layer12(out)
        out = self.layer7(out)


        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.


        return out



# class Unet(nn.Module):
#   def __init__(self, in_channels, middle_channels, out_channels):
#     super(Unet, self).__init__()
#     self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#     self.conv_relu = nn.Sequential(
#         nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
#         nn.ReLU(inplace=True)
#         )
#   def forward(self, x1, x2):
#     x1 = self.up(x1)
#     x1 = torch.cat((x1, x2), dim=1)
#     x1 = self.conv_relu(x1)
#     return x1
#
# class Decoder(nn.Module):
#     def __init__(self, n_class):
#         super().__init__()
#
#         self.base_model = torchvision.models.resnet18(True)
#         self.base_layers = list(self.base_model.children())
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#             self.base_layers[1],
#             self.base_layers[2])
#         self.layer2 = nn.Sequential(*self.base_layers[3:5])
#         self.layer3 = self.base_layers[5]
#         self.layer4 = self.base_layers[6]
#         self.layer5 = self.base_layers[7]
#         self.decode4 = Unet(512, 256+256, 256)
#         self.decode3 = Unet(256, 256+128, 256)
#         self.decode2 = Unet(256, 128+64, 128)
#         self.decode1 = Unet(128, 64+64, 64)
#         self.decode0 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
#             )
#         self.conv_last = nn.Conv2d(64, 1, kernel_size=3, padding=1)
#
#     def forward(self, image_with_wm):
#         e1 = self.layer1(image_with_wm) # 64,128,128
#         e2 = self.layer2(e1) # 64,64,64
#         e3 = self.layer3(e2) # 128,32,32
#         e4 = self.layer4(e3) # 256,16,16
#         f = self.layer5(e4) # 512,8,8
#         d4 = self.decode4(f, e4) # 256,16,16
#         d3 = self.decode3(d4, e3) # 256,32,32
#         d2 = self.decode2(d3, e2) # 128,64,64
#         d1 = self.decode1(d2, e1) # 64,128,128
#         d0 = self.decode0(d1) # 64,256,256
#         out = self.conv_last(d0) # 1,256,256
#         #print(out.shape)
#         return out