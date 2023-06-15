import torch
import torch.nn as nn
import os
import torch.utils.data
import cv2
import torchvision.transforms as transforms
import utils
import numpy as np


class MyData(torch.utils.data.Dataset):

    def __init__(self, images_folder, messages_folder):
        self.images = self.getImageList(images_folder)
        self.messages = self.getImageList(messages_folder)

    def __len__(self):
        # return len(self.images)

        return  len(self.images)



    def __getitem__(self, index):
        image_index = index
        message_index = index

        image = cv2.imread(self.images[image_index])


        image = cv2.resize(image, (256,256))
        # print(index)
        # print(self.images[image_index])
        # print(self.messages[message_index])
        with open('/new/zbb/HiDDeN-master/decoder_data/message0502.txt', 'a', encoding='utf-8') as f:
            f.write(self.messages[image_index]+'\n')
            f.close()
        with open('/new/zbb/HiDDeN-master/decoder_data/zaiti0502.txt', 'a', encoding='utf-8') as f:
            f.write(self.images[image_index]+'\n')
            f.close()

        image1 = image
        i = 0
        # if(i == 0):
        #     print(image[:,:,0])


        # print(image[2])
        # print(image.shape)
        # img_Ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # Y, Cb, Cr = cv2.split(img_Ycbcr)
        # print(np.min(Cb))


        # gray = cv2.cvtColor(img_Ycbcr, cv2.COLOR_BGR2GRAY)  # 将图像转为灰度图像
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 将图像进行高斯模糊
        # edge = cv2.Canny(gray, 1, 23)  # Canny边缘检测  0-255
        # weight = edge / 255.0
        #print(edge)
        #x, y = weight.shape
        # for m in range(x):
        #     for n in range(y):
        #         print(edge[m][n])


        trans = transforms.Compose([transforms.ToTensor()])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        image = trans(image)
        #print(image.shape)
        # image_np = image.detach().cpu().numpy()
        # #image_np = np.swapaxes(image_np, 2, 0)
        # image_np = np.transpose(image_np,(1,2,0))
        # #print(image_np.shape)
        # image_np = image_np*255.0
        # print(image_np - image1)
        # # image_np = np.array(image_np,dtype = np.int)
        # # if(i == 0):
        # #     print(image_np[:,:,0])
        # print('///////////////////////////')
        # # if(i == 0):
        # #     print(image_np[0])
        # i = i+1
        # img_Ycbcr_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCrCb)
        # Y_np,Cb_np,Cr_np = cv2.split(img_Ycbcr_np)
        # print(np.min(Cb_np))
        #print(image.shape) #[3,128,128]


        message = cv2.imread(self.messages[message_index], cv2.IMREAD_GRAYSCALE)

        #ret,thresh1 = cv2.threshold(message,115,255,cv2.THRESH_BINARY)       #转二值
        #print(message.shape) #28x28
        message = cv2.resize(message, (256,256))
        #print(message.shape)  #128x128
        #
        # message = message / 255.0 - (1 - weight)
        # x,y = message.shape
        # for i in range(x):
        #     for j in range(y):
        #         if(message[i][j] < 0):
        #             message[i][j] = 0
        # message_f = message.flattern()
        # for i in message_f:
        #     if message_f[i] <0:
        #         message_f[i] = 0
        # message = message_f.reshape((128,128))


        message = torch.as_tensor(message, dtype=torch.float)
        #print(message.shape)  #[128,128]
        message = torch.unsqueeze(message, dim=0)

        return image, message/255.0

    def getImageList(self, image_folder):
        image_names = os.listdir(image_folder)
        image_names.sort()
        # print(image_names)
        image_path = [os.path.join(image_folder, image_name) for image_name in image_names]
        return image_path


if __name__ == '__main__':
    images_folder = "/new/zbb/HiDDeN-master/decoder_data/encoded_data2"
    messages_folder = "/new/zbb/HiDDeN-master/decoder_data/message"
    test_images_folder = ""
    test_messages_folder = ""

    myData = MyData(images_folder, messages_folder)
    dataloader = torch.utils.data.dataloader.DataLoader(myData)


    img, mess = list(dataloader)[0]
   # print(img.shape)
   # print(mess.shape)

    for img, mess in dataloader:
        pass