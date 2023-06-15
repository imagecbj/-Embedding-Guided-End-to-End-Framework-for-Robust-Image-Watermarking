import torch
import torch.nn
import argparse
import os
import numpy as np
from options import HiDDenConfiguration

import utils
from model.hidden import Hidden
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF
import kornia

import SSIM

import cv2

import MakeDataset

#os.environ['CUDA_VISIBLE_DEVICES'] ='1'

# class NewDecoder(nn.Module):
#     """
#     Decoder module. Receives a watermarked image and extracts the watermark.
#     The input image may have various kinds of noise applied to it,
#     such as Crop, JpegCompression, and so on. See Noise layers for more.
#     """
#     def __init__(self, config: HiDDenConfiguration):
#
#         super(NewDecoder, self).__init__()
#
#
#         self.layer1 = conv_layer(3, 16, 3, 1)
#         self.layer2 = conv_layer(16, 32, 3, 1)
#         self.layer3 = conv_layer(32, 64, 3, 1)
#         self.layer4 = conv_layer(64, 128, 3, 1)
#         # self.layer5 = conv_layer(128, 256, 3, 1)
#         # self.layer6 = conv_layer(256, 512, 3, 1)
#         # self.layer7 = conv_layer(512, 256, 3, 1)
#         # self.layer8 = conv_layer(256, 128, 3, 1)
#         self.layer5 = conv_layer(128, 64, 3, 1)
#         self.layer6 = conv_layer(64, 32, 3, 1)
#         # self.layer11 = conv_layer(32, 3, 3, 1)
#         # self.layer12 = conv_layer(3, 1, 3, 1)
#         self.layer7 = conv_layer(32, 1, 3, 1)
#
#     def forward(self, image_with_wm):
#         out = self.layer1(image_with_wm)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         # out = self.layer5(out)
#         # out = self.layer6(out)
#         # out = self.layer7(out)
#         # out = self.layer8(out)
#         out = self.layer5(out)
#         out = self.layer6(out)
#         # out = self.layer11(out)
#         # out = self.layer12(out)
#         out = self.layer7(out)
#
#
#         # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
#         # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
#
#
#         return out
#

def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img

def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    #input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


def main():
    device = torch.device('cpu')
    #device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    #######################################
    print("using {} device.".format(device))

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    parser.add_argument('--source-image', '-s', required=True, type=str,
                        help='The image to watermark')
    # parser.add_argument('--times', '-t', default=10, type=int,
    #                     help='Number iterations (insert watermark->extract).')

    args = parser.parse_args()

    train_options, hidden_config, noise_config = utils.load_options(args.options_file)
    print('noise_config:',noise_config)
    noiser = Noiser(noise_config,device)

    checkpoint = torch.load(args.checkpoint_file,map_location='cpu')
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)
    ### 拆分decoder
    enmodel = hidden_net.encoder_decoder.encoder
    # for i in demodel.state_dict():
    #     print(i)
    #torch.save(demodel.state_dict(), 'net_params.pkl')

    images_folder = "/new/zbb/HiDDeN-master/coco_data/train"
    messages_folder = "/new/zbb/Data/image/train"
    train_dataset = MakeDataset.MyData(images_folder, messages_folder)
    test_data = torch.utils.data.dataloader.DataLoader(train_dataset, shuffle=True)  # shuffle=True
    i = 0
    
    for image,message in test_data:
        image = image.to(device)
        message = message.to(device)


        ProbabilityValue = enmodel(image, message)

        filename = os.path.join('/new/zbb/HiDDeN-master/decoder_data/encoded0502/', str(i)+'.png')
        with open('/new/zbb/HiDDeN-master/decoder_data/res0404.txt', 'a', encoding='utf-8') as f:
            f.write(filename+'\n')
            f.close()
        # print(filename)
        save_image_tensor2cv2(ProbabilityValue, filename)
        i = i+1



    # image = cv2.imread('test.png')
    # image = cv2.resize(image, (128, 128))
    # image = image/255.0
    # print('image12:', image.shape)
    # image = np.transpose(image, (2, 0, 1))
    # image = np.array(image, np.float32)
    # image = torch.tensor(image, dtype=torch.float32)
    # image = torch.unsqueeze(image, dim=0)
    # print('image1:',image.shape)
    # 
    # message = cv2.imread('message.png')
    # message = cv2.resize(message, (128, 128))
    # message = cv2.cvtColor(message, cv2.COLOR_BGR2GRAY)
    # message = message / 255.0
    # print('message1:',message.shape)
    # #message = np.transpose(message, (2, 0, 1))
    # message = np.array(message, np.float32)
    # message = torch.tensor(message, dtype=torch.float32)
    # message = torch.unsqueeze(message, dim=0)
    # message = torch.unsqueeze(message, dim=0)
    # print('message2:', message.shape)

    # image = cv2.imread('test.png')
    # image = cv2.resize(image, (128, 128))
    # image = image/255.0
    # image = np.transpose(image, (2, 0, 1))
    # image = np.array(image, np.float32)
    # image = torch.tensor(image, dtype=torch.float32)
    # image = torch.unsqueeze(image, dim=0)
    #
    # message = cv2.imread('message.png')

    # print(image.shape)
    # #image = torch.unsqueeze(image, dim=0)
    # ProbabilityValue = enmodel(image,message)
    # print('probablity:',ProbabilityValue.shape)
    # save_image_tensor2cv2(image,'/new/zbb/HiDDeN-master/testdecoder/new/image.png')
    # save_image_tensor2cv2(ProbabilityValue, '/new/zbb/HiDDeN-master/testdecoder/new/pro.png')
    #utils.save_images(image.cpu(), ProbabilityValue.cpu(), 'test', 'testdecoder2', resize_to=(128, 128))





    # images_folder = "/new/zbb/HiDDeN-master/coco_data/test"
    # messages_folder = "/new/zbb/Data/logo/test"
    # train_dataset = MakeDataset.MyData(images_folder, messages_folder)
    # test_data = torch.utils.data.dataloader.DataLoader(train_dataset,shuffle=True)  # shuffle=True
    #
    #
    #
    # # for t in range(args.times):
    # test_result = {
    #
    #     "error1": 0.0,
    #     "error2": 0.0,
    #     "psnr": 0.0,
    #     "psnr1": 0.0,
    #     "ssim": 0.0,
    #     "ssim2": 0.0,
    # }
    #
    # num = 0
    #
    # for image, message in test_data:
    #     image = image.to(device)
    #
    #     message = message.to(device)
    #     print('message:',message.shape)
    #
    #
    #     # image encoded_images
    #     # message decoded_messages
    #
    #     losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([image, message])
    #     #decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
    #     decoded_rounded = decoded_messages.detach().cpu().numpy()
    #     message_detached = message.detach().cpu().numpy()
    #     # print("message: ",message_detached.shape)
    #     # print("decoded: ", decoded_rounded.shape)
    #     # print('original: {}'.format(message_detached))
    #     # print('decoded : {}'.format(decoded_rounded))
    #
    #     # psnr
    #
    #     psnr = kornia.losses.psnr_loss(image,encoded_images.detach(), 2).item()
    #     psnr2 = kornia.losses.psnr_loss(message, decoded_messages.detach(), 2).item()
    #     print("psnr: ",-psnr)
    #     print("psnr2: ", -psnr2)
    #
    #     # ssim
    #     # print(encoded_images.shape)
    #     # print(image.shape)
    #
    #     K = [0.01, 0.03]
    #     L = 128;
    #     window_size = 11
    #     ssim = SSIM.ssim(encoded_images.detach(),image,K,window_size,L)
    #     ssim2 = SSIM.ssim(decoded_messages.detach(), message, K, window_size, L)
    #     print ("ssim1: ",ssim)
    #     print("ssim2: ", ssim2)
    #
    #     # test = kornia.losses.ssim(encoded_images.detach(), image, window_size=5).item()
    #     # ssim = 1 - 2 * kornia.losses.ssim(encoded_images.detach(), image, window_size=5).item()
    #     error = np.mean(np.abs(decoded_rounded - message_detached))
    #     print('error : {:.3f}'.format(np.mean(np.abs(decoded_rounded - message_detached))))
    #
    #     error2 = np.mean(np.abs(image.detach().cpu().numpy() - encoded_images.detach().cpu().numpy()))
    #     #print('error2 : {:.3f}'.format(np.mean(np.abs(image - encoded_images))))
    #     print('error2: ',error2)
    #
    #
    #     #error.append(np.mean(np.abs(decoded_rounded - message_detached)))
    #
    #     result = {
    #         "error1": error,
    #         "error2": error2,
    #         "psnr": psnr,
    #         "psnr1": psnr2,
    #         "ssim": ssim,
    #         "ssim2": ssim2,
    #     }
    #
    #     for key in result:
    #         test_result[key] += float(result[key])
    #
    #     utils.save_images(image.cpu(), encoded_images.cpu(), 'test','encoded', resize_to=(128,128))
    #     utils.save_images(message.cpu(), decoded_messages.cpu(), 'test','message', resize_to=(128,128))
    #
    #
    #     num = num+1
    #
    # # sum = 0
    # # _min = 1
    # # for x in error:
    # #     sum = sum + x
    # #     _min = min(_min,x)
    # # print("平均值： ",sum/len(error))
    # # print("最小值： ",_min)
    #
    # '''
    # test results
    # '''
    # content = "Average : \n"
    # for key in test_result:
    #     content += key + "=" + str(test_result[key] / num) + ","
    # content += "\n"
    #
    # # with open(test_log, "a") as file:
    # #     file.write(content)
    #
    # print(content)
    # # bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy()))/(image_tensor.shape[0] * messages.shape[1])
    #


if __name__ == '__main__':
    main()
