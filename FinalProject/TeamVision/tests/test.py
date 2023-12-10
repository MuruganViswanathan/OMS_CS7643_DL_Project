"""
Infer segmentation results from a trained SegNet model


Usage:
python inference.py --data_root /home/SharedData/intern_sayan/PascalVOC2012/data/VOCdevkit/VOC2012/ \
                    --val_path ImageSets/Segmentation/val.txt \
                    --img_dir JPEGImages \
                    --mask_dir SegmentationClass \
                    --model_path /home/SharedData/intern_sayan/PascalVOC2012/model_best.pth \
                    --output_dir /home/SharedData/intern_sayan/PascalVOC2012/predictions \
                    --gpu 1
"""

import os
from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
from ..data.dataset import PascalVOCDataset, NUM_CLASSES
from ..models.model import SegNet
from ..main import get_model_params, get_training_params, get_data_params
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

plt.switch_backend('agg')
plt.axis('off')


# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = NUM_CLASSES


model_name, save_best, checkpoint_file, out_dir = get_model_params()
batch_size, epochs, learning_rate, regularization, momentum = get_training_params()
dataset_name, dataset_path, train_data_file, val_data_file, img_dir, mask_dir = get_data_params()


# Arguments
parser = argparse.ArgumentParser(description='Validate a SegNet model')
parser.add_argument('--gpu', type=int)
args = parser.parse_args()


def validate():
    model.eval()

    for batch_idx, batch in enumerate(val_dataloader):
        input_tensor = torch.autograd.Variable(batch['image'])
        target_tensor = torch.autograd.Variable(batch['mask'])

        if CUDA:
            input_tensor = input_tensor.cuda(GPU_ID)
            target_tensor = target_tensor.cuda(GPU_ID)

        predicted_tensor, softmaxed_tensor = model(input_tensor)
        loss = criterion(predicted_tensor, target_tensor)

        for idx, predicted_mask in enumerate(softmaxed_tensor):
            target_mask = target_tensor[idx]
            input_image = input_tensor[idx]

            fig = plt.figure()

            a = fig.add_subplot(1,3,1)
            plt.imshow(input_image.transpose(0, 2))
            a.set_title('Input Image')

            a = fig.add_subplot(1,3,2)
            predicted_mx = predicted_mask.detach().cpu().numpy()
            predicted_mx = predicted_mx.argmax(axis=0)
            plt.imshow(predicted_mx)
            a.set_title('Predicted Mask')

            a = fig.add_subplot(1,3,3)
            target_mx = target_mask.detach().cpu().numpy()
            plt.imshow(target_mx)
            a.set_title('Ground Truth')

            fig.savefig(os.path.join(out_dir, "prediction_{}_{}.png".format(batch_idx, idx)))

            plt.close(fig)


if __name__ == "__main__":

    CUDA = args.gpu is not None
    GPU_ID = args.gpu

    val_dataset = PascalVOCDataset(list_file=val_data_file,
                                   img_dir=img_dir,
                                   mask_dir=mask_dir)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)


    if CUDA:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS).cuda(GPU_ID)

        class_weights = 1.0/val_dataset.get_class_probability().cuda(GPU_ID)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)
    else:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS)

        class_weights = 1.0/val_dataset.get_class_probability()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)


    model.load_state_dict(torch.load(checkpoint_file))


    validate()


