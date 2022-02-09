"""
UnetDownward model for self-supervised learning
It is the encoder of the UNet_3Plus from this repository

also returns an OrderedDict; index the last entry for the final output, or "hd5"

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from unet.models.layers import unetConv2


class UnetDownward(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True, get_intermediate_outputs=False):
        super(UnetDownward, self).__init__()
        filters = [64, 128, 256, 512, 1024]

        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.out_channels = 1024 if not get_intermediate_outputs else sum(filters)
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        self.get_intermediate_outputs = get_intermediate_outputs


        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)


    def forward(self, inputs):

        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        result = OrderedDict()
        if self.get_intermediate_outputs:
            result['h1'] = h1
            result['h2'] = h2
            result['h3'] = h3
            result['h4'] = h4
        result['hd5'] = hd5

        return result