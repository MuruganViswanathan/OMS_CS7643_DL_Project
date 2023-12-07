"""
Pytorch implementation of SegNet (https://arxiv.org/pdf/1511.00561.pdf)

Notes:
    #1 Fig.3.The encoder network consists of 13 convolutional layers which correspond to the first 13 convolutional
    layers in the VGG16 network[1] designed for object classification.
    We can therefore initialize the training process from weights trained for classification on large datasets [41].

"""

import torch
import torch.nn as nn
import torchvision.models as models

DEBUG = False

class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.num_channels = input_channels

        # VGG16
        # initialize the training process from weights trained for classification on large datasets
        self.vgg16 = models.vgg16(weights='DEFAULT')
        self.init_vgg_weigts()

        # Encoder layers
        self.encoder = self.build_encoder()

        # Decoder layers
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder = nn.Sequential(
            self.build_encoder_layer(self.input_channels, 64),
            self.build_encoder_layer(64, 128),
            self.build_encoder_layer(128, 256),
            self.build_encoder_layer(256, 512),
            self.build_encoder_layer(512, 512),
        )
        return encoder

    #
    # encoder_layer:
    #   Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool
    #
    def build_encoder_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )

    def build_decoder(self):
        decoder = nn.Sequential(
            self.build_decoder_layer(512, 512),
            self.build_decoder_layer(512, 512),
            self.build_decoder_layer(512, 512),
            self.build_decoder_layer(512, 512),
            self.build_decoder_layer(512, 256),
            self.build_decoder_layer(256, 128),
            self.build_decoder_layer(128, 64),
        )
        return decoder

    #
    # decoder_layer:
    #   MaxUnpool -> Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    #
    def build_decoder_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, input_img):
        # Encoder
        encoded, indices = [], []
        x = input_img
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                x, idx = layer(x)
                encoded.append(x)
                indices.append(idx)
            else:
                x = layer(x)

        # Decoder
        for layer in self.decoder:
            if isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, indices.pop(), output_size=encoded.pop().size())
            else:
                x = layer(x)

        x_softmax = nn.functional.softmax(x, dim=1)

        if DEBUG:
            for i, (enc, dec) in enumerate(zip(encoded, reversed(self.decoder))):
                print(f"dim_{i}: {enc.size()} -> {dec[0].weight.size(0)} channels")

        return x, x_softmax


    def init_vgg_weigts(self):
        assert self.encoder_conv_00[0].weight.size() == self.vgg16.features[0].weight.size()
        self.encoder_conv_00[0].weight.data = self.vgg16.features[0].weight.data
        assert self.encoder_conv_00[0].bias.size() == self.vgg16.features[0].bias.size()
        self.encoder_conv_00[0].bias.data = self.vgg16.features[0].bias.data

        assert self.encoder_conv_01[0].weight.size() == self.vgg16.features[2].weight.size()
        self.encoder_conv_01[0].weight.data = self.vgg16.features[2].weight.data
        assert self.encoder_conv_01[0].bias.size() == self.vgg16.features[2].bias.size()
        self.encoder_conv_01[0].bias.data = self.vgg16.features[2].bias.data

        assert self.encoder_conv_10[0].weight.size() == self.vgg16.features[5].weight.size()
        self.encoder_conv_10[0].weight.data = self.vgg16.features[5].weight.data
        assert self.encoder_conv_10[0].bias.size() == self.vgg16.features[5].bias.size()
        self.encoder_conv_10[0].bias.data = self.vgg16.features[5].bias.data

        assert self.encoder_conv_11[0].weight.size() == self.vgg16.features[7].weight.size()
        self.encoder_conv_11[0].weight.data = self.vgg16.features[7].weight.data
        assert self.encoder_conv_11[0].bias.size() == self.vgg16.features[7].bias.size()
        self.encoder_conv_11[0].bias.data = self.vgg16.features[7].bias.data

        assert self.encoder_conv_20[0].weight.size() == self.vgg16.features[10].weight.size()
        self.encoder_conv_20[0].weight.data = self.vgg16.features[10].weight.data
        assert self.encoder_conv_20[0].bias.size() == self.vgg16.features[10].bias.size()
        self.encoder_conv_20[0].bias.data = self.vgg16.features[10].bias.data

        assert self.encoder_conv_21[0].weight.size() == self.vgg16.features[12].weight.size()
        self.encoder_conv_21[0].weight.data = self.vgg16.features[12].weight.data
        assert self.encoder_conv_21[0].bias.size() == self.vgg16.features[12].bias.size()
        self.encoder_conv_21[0].bias.data = self.vgg16.features[12].bias.data

        assert self.encoder_conv_22[0].weight.size() == self.vgg16.features[14].weight.size()
        self.encoder_conv_22[0].weight.data = self.vgg16.features[14].weight.data
        assert self.encoder_conv_22[0].bias.size() == self.vgg16.features[14].bias.size()
        self.encoder_conv_22[0].bias.data = self.vgg16.features[14].bias.data

        assert self.encoder_conv_30[0].weight.size() == self.vgg16.features[17].weight.size()
        self.encoder_conv_30[0].weight.data = self.vgg16.features[17].weight.data
        assert self.encoder_conv_30[0].bias.size() == self.vgg16.features[17].bias.size()
        self.encoder_conv_30[0].bias.data = self.vgg16.features[17].bias.data

        assert self.encoder_conv_31[0].weight.size() == self.vgg16.features[19].weight.size()
        self.encoder_conv_31[0].weight.data = self.vgg16.features[19].weight.data
        assert self.encoder_conv_31[0].bias.size() == self.vgg16.features[19].bias.size()
        self.encoder_conv_31[0].bias.data = self.vgg16.features[19].bias.data

        assert self.encoder_conv_32[0].weight.size() == self.vgg16.features[21].weight.size()
        self.encoder_conv_32[0].weight.data = self.vgg16.features[21].weight.data
        assert self.encoder_conv_32[0].bias.size() == self.vgg16.features[21].bias.size()
        self.encoder_conv_32[0].bias.data = self.vgg16.features[21].bias.data

        assert self.encoder_conv_40[0].weight.size() == self.vgg16.features[24].weight.size()
        self.encoder_conv_40[0].weight.data = self.vgg16.features[24].weight.data
        assert self.encoder_conv_40[0].bias.size() == self.vgg16.features[24].bias.size()
        self.encoder_conv_40[0].bias.data = self.vgg16.features[24].bias.data

        assert self.encoder_conv_41[0].weight.size() == self.vgg16.features[26].weight.size()
        self.encoder_conv_41[0].weight.data = self.vgg16.features[26].weight.data
        assert self.encoder_conv_41[0].bias.size() == self.vgg16.features[26].bias.size()
        self.encoder_conv_41[0].bias.data = self.vgg16.features[26].bias.data

        assert self.encoder_conv_42[0].weight.size() == self.vgg16.features[28].weight.size()
        self.encoder_conv_42[0].weight.data = self.vgg16.features[28].weight.data
        assert self.encoder_conv_42[0].bias.size() == self.vgg16.features[28].bias.size()
        self.encoder_conv_42[0].bias.data = self.vgg16.features[28].bias.data

