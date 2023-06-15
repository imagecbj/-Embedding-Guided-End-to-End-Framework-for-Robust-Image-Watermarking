import torch.nn as nn
import sys
import torchvision
import torch
import torchvision.transforms as transforms
import MakeDataset
import os
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

savepath = '/new/zbb/HiDDeN-master/decoder_model/'

LEARNING_RATE = 0.0001
EPOCH = 3000
BATCH_SIZE = 8


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
      transforms.Resize([64,64]),
      transforms.ToTensor(),
])

# trainData = dsets.ImageFolder('/data1/celeba_df/train/',transform)
# testData = dsets.ImageFolder('/data1/FF++faceonly/newtrain/testneu/',transform)
#
# trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
# testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

images_folder = "/new/zbb/HiDDeN-master/decoder_data/encoded0502"
messages_folder = "/new/zbb/HiDDeN-master/decoder_data/message0502"
train_dataset = MakeDataset.MyData(images_folder, messages_folder)
train_data = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)  # shuffle=True

val_images_folder = "/new/zbb/HiDDeN-master/decoder_data/val/encoded_data"
val_messages_folder = "/new/zbb/HiDDeN-master/decoder_data/val/message"
val_dataset = MakeDataset.MyData(val_images_folder, val_messages_folder)
val_data = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size=BATCH_SIZE)


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
    def __init__(self):
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
        return out
                # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
                # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.


def decoder_model(pretrained=False,**kwargs):
    """
    Construct Xception.
    """

    model = Decoder(**kwargs)

    return model


def train():
    decoderModel = decoder_model()

    # model_weight_path = "/new/zbb/HiDDeN-master/net_params.pkl"
    # decoderModel.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    use_gpu = torch.cuda.is_available()
    #use_gpu = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    if use_gpu:
        decoderModel.cuda()
    # decoderModel.load_state_dict(torch.load('40cnn.pkl',map_location='cpu'))


    cost = nn.MSELoss(reduce=True, size_average=True)
    optimizer = torch.optim.Adam(decoderModel.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Train the model
    for epoch in range(EPOCH):

        avg_loss = 0
        cnt = 0

        test_avgloss = 0
        test_cnt = 0
        for images, messages in train_data:
            if use_gpu:
                images = images.cuda()
                messages = messages.cuda()

            optimizer.zero_grad()
            outputs = decoderModel(images)

            loss = cost(outputs, messages)
            # print(loss.data)
            avg_loss += loss.data
            cnt += 1
            print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss / cnt))
            loss.backward()
            optimizer.step()
        scheduler.step(avg_loss)
        #验证集
        index = 0
        for images, messages in val_data:
            if use_gpu:
                images = images.cuda()
                messages = messages.cuda()

            # Forward + Backward + Optimize
            #print(images)
            optimizer.zero_grad()
            outputs = decoderModel(images)

            #filename = os.path.join('/new/zbb/robust_data/encoded_data2/', str(epoch)+'-'+str(index) + '.png')
            # save_image_tensor2cv2(outputs, filename)
            utils.save_images(messages.cpu(), outputs.cpu(), 'test'+str(epoch), 'message_pretrain', resize_to=(64,64))
            index = index + 1
        torch.save(decoderModel.state_dict(), savepath + str(epoch) + 'decoder.pkl')

def test():
    pass

if __name__ == '__main__':
    train()

