"""
Train a SegNet model

Usage:
python train.py --gpu 0
"""

import os
import time
import torch
from torch.utils.data import DataLoader
import argparse
from data.dataset import PascalVOCDataset, NUM_CLASSES
from models.model import SegNet
from main import get_model_params, get_training_params, get_data_params

# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = NUM_CLASSES

model_name, save_best, checkpoint_file, out_dir = get_model_params()
batch_size, epochs, learning_rate, regularization, momentum = get_training_params()
dataset_name, dataset_path, train_data_file, val_data_file, img_dir, mask_dir = get_data_params()

# Arguments
parser = argparse.ArgumentParser(description='Train my SegNet model')
parser.add_argument('--gpu', type=int)
args = parser.parse_args()


def train(model, train_dataloader, criterion, optimizer, checkpoint_file):
    is_better = True
    prev_loss = float('inf')

    model.train()

    for epoch in range(epochs):
        loss_f = 0
        t_start = time.time()

        for batch in train_dataloader:
            input_tensor = batch['image']
            target_tensor = batch['mask']

            if CUDA:
                input_tensor = input_tensor.cuda(GPU_ID)
                target_tensor = target_tensor.cuda(GPU_ID)

            input_var = torch.autograd.Variable(input_tensor)
            target_var = torch.autograd.Variable(target_tensor)

            if CUDA:
                input_var = input_var.cuda(GPU_ID)
                target_var = target_var.cuda(GPU_ID)

            predicted_tensor, softmaxed_tensor = model(input_var)

            optimizer.zero_grad()
            loss = criterion(predicted_tensor, target_var)
            loss.backward()
            optimizer.step()

            loss_f += loss.item()
            prediction_f = softmaxed_tensor.float()

        delta = time.time() - t_start
        is_better = loss_f < prev_loss

        if is_better:
            prev_loss = loss_f
            torch.save(model.state_dict(), checkpoint_file)

        print("Epoch #{}\tLoss: {:.8f}\t Time: {:.2f}s".format(epoch + 1, loss_f, delta))


if __name__ == "__main__":
    CUDA = args.gpu is not None
    GPU_ID = args.gpu

    train_dataset = PascalVOCDataset(
        list_file=train_data_file,
        img_dir=img_dir,
        mask_dir=mask_dir
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    if CUDA:
        model = SegNet(
            input_channels=NUM_INPUT_CHANNELS,
            output_channels=NUM_OUTPUT_CHANNELS
        ).cuda(GPU_ID)
        class_weights = 1.0 / train_dataset.get_class_probability().cuda(GPU_ID)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)
    else:
        model = SegNet(
            input_channels=NUM_INPUT_CHANNELS,
            output_channels=NUM_OUTPUT_CHANNELS
        )
        class_weights = 1.0 / train_dataset.get_class_probability()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    if os.path.exists(checkpoint_file):
        model.load_state_dict(torch.load(checkpoint_file))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_dataloader, criterion, optimizer, checkpoint_file)
