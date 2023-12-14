"""
References:
    1) Dataset: https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012/
    2) SegNet: https://arxiv.org/pdf/1511.00561.pdf
    3) Referencing various Semantic Segmentation models implementations:
        https://github.com/Tramac/awesome-semantic-segmentation-pytorch
        https://github.com/say4n/pytorch-segnet
    4) Original paper: https://doi.org/10.48550/arXiv.1511.00561

Authors (listed alphabetically):
    Murugan Viswanathan
    Paresh Upadhyay
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
        self.vgg16_pretrained = models.vgg16(weights = 'DEFAULT')

        # Encoder layers and load weights and parameters from vgg16 pretrained model
        self.encoder = self.build_encoder()
        self.encoder.load_state_dict(self.vgg16_pretrained.state_dict(), strict=False)

        # Decoder layers
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder = nn.Sequential(
            self.build_encoder_layer(self.input_channels, 64),
            self.build_encoder_layer(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            self.build_encoder_layer(64, 128),
            self.build_encoder_layer(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            self.build_encoder_layer(128, 256),
            self.build_encoder_layer(256, 256),
            self.build_encoder_layer(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            self.build_encoder_layer(256, 512),
            self.build_encoder_layer(512, 512),
            self.build_encoder_layer(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            self.build_encoder_layer(512, 512),
            self.build_encoder_layer(512, 512),
            self.build_encoder_layer(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        )
        return encoder

    #
    # encoder_layer:
    #   Conv -> BatchNorm -> ReLU
    #
    def build_encoder_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def build_decoder(self):
        decoder = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            self.build_decoder_layer(512, 512),
            self.build_decoder_layer(512, 512),
            self.build_decoder_layer(512, 512),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            self.build_decoder_layer(512, 512),
            self.build_decoder_layer(512, 512),
            self.build_decoder_layer(512, 256),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            self.build_decoder_layer(256, 256),
            self.build_decoder_layer(256, 256),
            self.build_decoder_layer(256, 128),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            self.build_decoder_layer(128, 128),
            self.build_decoder_layer(128, 64),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            self.build_decoder_layer(64, 64),
            self.build_decoder_layer(64, self.output_channels)

        )
        return decoder

    #
    # decoder_layer:
    #   Conv -> BatchNorm -> ReLU
    #
    def build_decoder_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input_img):
        # Encoder
        encoded, indices = [], []
        x = input_img
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                encoded.append(x)
                x, idx = layer(x)
                indices.append(idx)
            else:
                x = layer(x)
        # Decoder
        for layer in self.decoder:

            if isinstance(layer, nn.MaxUnpool2d):
                out_size = encoded.pop().size()
                x = layer(x, indices.pop(), output_size=out_size)
            else:
                x = layer(x)
        x_softmax = nn.functional.softmax(x, dim=1)

        if DEBUG:
            for i, (enc, dec) in enumerate(zip(encoded, reversed(self.decoder))):
                print(f"dim_{i}: {enc.size()} -> {dec[0].weight.size(0)} channels")

        return x, x_softmax

