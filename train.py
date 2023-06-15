
import os
import time
import torch
import numpy as np
import utils
import logging
from collections import defaultdict
from PIL import Image

from options import *
from model.hidden import Hidden
from average_meter import AverageMeter

import torchvision
from torchvision import models
from torch import Tensor
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import cv2
import MakeDataset
import torch.utils.data.dataloader




transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])


# message = dsets.ImageFolder('/new/zbb/HiDDeN-master/mnist/train1/', transform=transform)
# message_loader = torch.utils.data.DataLoader(message, batch_size=train_options.batch_size, shuffle=True,
#                                             num_workers=4)



def train(model: Hidden,
          device: torch.device,
          hidden_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger):
    """
    Trains the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param hidden_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """


    train_data , val_data = utils.get_data_loaders(hidden_config, train_options)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 100
    images_to_save = 8
    batch_size = 8
    saved_images_size = (256,256)

    images_folder = "/new/zbb/HiDDeN-master/coco_data/train"
    messages_folder = "/new/zbb/Data/image/train"
    train_dataset = MakeDataset.MyData(images_folder, messages_folder)
    train_data = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)  #shuffle=True

    val_images_folder = "/new/zbb/HiDDeN-master/coco_data/val"
    val_messages_folder = "/new/zbb/Data/image/test"
    val_dataset = MakeDataset.MyData(val_images_folder, val_messages_folder)
    val_data = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        for image, message in train_data:
            image = image.to(device)
            message = message.to(device)
            # message = dsets.ImageFolder('/new/zbb/HiDDeN-master/coco_data/', transform=transform)
            # train_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size, shuffle=True,
            #                                            num_workers=4)

            losses, _= model.train_on_batch([image, message])

            for name, loss in losses.items():
                training_losses[name].update(loss)
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.log_progress(training_losses)
                logging.info('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)
        if tb_logger is not None:
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        first_iteration = True
        validation_losses = defaultdict(AverageMeter)
        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        for image, message in val_data:
            image = image.to(device)
            message = message.to(device)

            losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([image, message])
            for name, loss in losses.items():
                validation_losses[name].update(loss)
            if first_iteration:
                if hidden_config.enable_fp16:
                    image = image.float()
                    encoded_images= encoded_images.float()
                utils.save_images(image.cpu()[:images_to_save, :, :, :],
                                  encoded_images[:images_to_save, :, :, :].cpu(),
                                  epoch,
                                  os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                # utils.save_image(message.cpu()[:images_to_save, :, :, :],
                #                   decoded_messages[:images_to_save, :, :, :].cpu(),
                #                   epoch,
                #                   os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                first_iteration = False

        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)


