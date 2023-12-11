# train_validate.py

import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from data.dataset import PascalVOCDataset, NUM_CLASSES
from models.model import SegNet
from utils.visualization import visualize_predictions
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
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


# Function to visualize predictions
# def visualize_predictions(input_image, predicted_mask, target_mask, out_dir, idx, batch_idx):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
#     axes[0].imshow(input_image.permute(1, 2, 0))
#     axes[0].set_title('Input Image')
#
#     axes[1].imshow(predicted_mask, cmap='jet')
#     axes[1].set_title('Predicted Mask')
#
#     axes[2].imshow(target_mask, cmap='jet')
#     axes[2].set_title('Ground Truth')
#
#     save_path = os.path.join(out_dir, f"prediction_{batch_idx}_{idx}.png")
#     plt.savefig(save_path)
#     plt.close(fig)
#     return save_path

# Function to calculate mIoU
def calculate_miou(conf_matrix):
    intersection = np.diag(conf_matrix)
    union = np.sum(conf_matrix, axis=0) + np.sum(conf_matrix, axis=1) - np.diag(conf_matrix)
    iou = intersection / union
    mIoU = np.nanmean(iou)
    return mIoU


def train_and_validate(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, checkpoint_file,
                       out_dir):
    is_better = True
    prev_loss = float('inf')
    best_loss = float('inf')

    train_losses = []
    val_losses = []
    accuracies = []
    mious = []

    model.train()

    for epoch in range(epochs):
        curr_loss = 0
        start_time = time.time()

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

            predicted_tensor, _ = model(input_var)

            optimizer.zero_grad()
            loss = criterion(predicted_tensor, target_var)
            loss.backward()
            optimizer.step()

            curr_loss += loss.item()

        train_losses.append(curr_loss)
        scheduler.step()
        delta = time.time() - start_time
        # is_better = curr_loss < prev_loss
        is_best = curr_loss < best_loss

        print("Epoch #{}/{}\tLoss: {:.8f}\t Time: {:.2f}s".format(epoch + 1, epochs, curr_loss, delta))

        # Save the best model
        if is_best:
            best_loss = curr_loss
            checkpoint_info = {
                'epoch': epoch + 1,
                'loss': curr_loss,
            }

            print("save best model... Loss:", best_loss)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'checkpoint_info': checkpoint_info
            }, checkpoint_file)


        # Validation phase
        model.eval()
        val_loss = 0
        all_predicted_labels = []
        all_target_labels = []

        for batch_idx, batch in enumerate(val_dataloader):
            input_tensor = batch['image']
            target_tensor = batch['mask']

            if CUDA:
                input_tensor = input_tensor.cuda(GPU_ID)
                target_tensor = target_tensor.cuda(GPU_ID)

            with torch.no_grad():
                predicted_tensor, _ = model(input_tensor)
                loss = criterion(predicted_tensor, target_tensor)
                val_loss += loss.item()

                # Calculate accuracy
                # predicted label is index of the maximum values along dim 1
                _, predicted_labels = torch.max(predicted_tensor, 1)
                # all_predicted_labels.extend(predicted_labels.cpu().numpy())
                # all_target_labels.extend(target_tensor.cpu().numpy())

                all_target_labels = [tuple(label) for label in all_target_labels]
                all_predicted_labels = [tuple(label) for label in all_predicted_labels]

        val_losses.append(val_loss)

        print("Unique target labels:", set(all_target_labels))
        print("Unique predicted labels:", set(all_predicted_labels))

        accuracy = accuracy_score(all_target_labels, all_predicted_labels)
        print("Accuracy:", accuracy)
        accuracies.append(accuracy)

        # Calculate mIoU
        conf_matrix = confusion_matrix(all_target_labels, all_predicted_labels)
        mIoU = calculate_miou(conf_matrix)
        mious.append(mIoU)

        print("Validation Loss: {:.8f}\t Accuracy: {:.4f}\t mIoU: {:.4f}".format(val_loss, accuracy, mIoU))

        # Visualize some predictions
        visualize_predictions(input_tensor[0], predicted_tensor[0].argmax(dim=0).cpu().numpy(),
                              target_tensor[0].cpu().numpy(), out_dir, 0, epoch)

    # Plot loss curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'))

    # Plot accuracy curve
    plt.figure()
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy_curve.png'))

    # Plot mIoU curve
    plt.figure()
    plt.plot(mious, label='mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'miou_curve.png'))


if __name__ == "__main__":

    CUDA = args.gpu is not None
    GPU_ID = args.gpu

    print("Loading training PascalVOCDataset...")
    train_dataset = PascalVOCDataset(
        list_file=train_data_file,
        img_dir=img_dir,
        mask_dir=mask_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    print("Training DataLoader...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    print("Loading validation PascalVOCDataset...")
    val_dataset = PascalVOCDataset(list_file=val_data_file, img_dir=img_dir, mask_dir=mask_dir)

    print("Validation DataLoader...")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    if CUDA:
        print("CUDA True...")
        print("Loading model SegNet...")
        model = SegNet(
            input_channels=NUM_INPUT_CHANNELS,
            output_channels=NUM_OUTPUT_CHANNELS
        ).cuda(GPU_ID)

        class_weights = 1.0 / train_dataset.get_class_probability().cuda(GPU_ID)
        criterion = CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)

    else:
        print("CUDA False...")
        print("Loading model SegNet...")
        model = SegNet(
            input_channels=NUM_INPUT_CHANNELS,
            output_channels=NUM_OUTPUT_CHANNELS
        )

        class_weights = 1.0 / train_dataset.get_class_probability()
        criterion = CrossEntropyLoss(weight=class_weights)

    print("optimizer...")
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    if os.path.exists(checkpoint_file):
        print("Loading checkpoint_file...")
        checkpoint = torch.load(checkpoint_file)

        # Check if 'model_state_dict' key is present, otherwise assume direct state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        # Load optimizer state_dict if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Optimizer state_dict not found in the checkpoint file.")

        # Load scheduler state_dict if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("Scheduler state_dict not found in the checkpoint file.")

    else:
        print("checkpoint_file not found...")


    print("train and validate model...")
    train_and_validate(model, train_dataloader, val_dataloader, criterion, optimizer,
                       scheduler, checkpoint_file, out_dir)
