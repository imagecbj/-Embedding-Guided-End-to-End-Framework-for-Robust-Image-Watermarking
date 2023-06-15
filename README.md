# -Embedding-Guided-End-to-End-Framework-for-Robust-Image-Watermarking
## Prerequisites
#### Ubuntu 18.04
#### NVIDIA GPU+CUDA CuDNN (CPU mode may also work, but untested)
#### Install Torch 1.3.1, torchvision 0.4.2 and dependencies

## Data
#### The dataset directory is located on the old server's "/new/zbb/coco_data/". This is an open source datas.

## Training and Test Details
#### When you train a model, you should change the input in train.py. The corresponding part should also be modified during testing.decoderTest.py can test the part of decoder and the decoderTrain.py can train the decoder net. If you want to modify the model, you can make the changes in the model folder.If you want to modify the noise layer, you can do so in Noise_ Make modifications in the layers folder.


## Related Works
#### [1] Zhu J, Kaplan R, Johnson J, et al. "Hidden: Hiding data with deep networks." Proceedings of the European conference on computer vision (ECCV). 2018.
