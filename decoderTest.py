import sys
import os
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
import datetime
import MakeDataset
import decoderTrain
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

savePath = "/new/zbb/HiDDeN-master/decoder_test"

def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # # 去掉批次维度
    #     # input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    #input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)

transform = transforms.Compose([
      #convertYcbcr(),
      transforms.Resize([128,128]),
      transforms.ToTensor(),
])


def func():

    device = torch.device("cpu")

    # # load image

    # # create model
    model = decoderTrain.decoder_model()
    # # load model weights
    model_weight_path = "/new/zbb/HiDDeN-master/decoder_model/3411decoder.pkl"
    model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    images_folder = "/new/zbb/robust_data/encoded_data2"
    messages_folder = "/new/zbb/robust_data/encoded_data2"

    test_dataset = MakeDataset.MyData(images_folder,messages_folder)
    test_data = torch.utils.data.dataloader.DataLoader(test_dataset, shuffle=True)  # shuffle=True
    i = 0

    for image,message in test_data:
        print(image)
        # images = torch.stack(image, dim=0)

        image = image.to(device)


        ProbabilityValue = model(image)

        filename = os.path.join( "/new/zbb/HiDDeN-master/decoder_test")
        # with open('/new/zbb/HiDDeN-master/decoder_data/res.txt', 'a', encoding='utf-8') as f:
        #     f.write(filename+'\n')
        #     f.close()
        # print(filename)
        #save_image_tensor2cv2(ProbabilityValue, filename)
        utils.save_images(ProbabilityValue.cpu(), ProbabilityValue.cpu(), 'test' + str(i), filename, resize_to=(128, 128))
        i = i + 1


if __name__ == '__main__':
    func()