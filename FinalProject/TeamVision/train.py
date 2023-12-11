import os
import time
import torch
from torch.utils.data import DataLoader
import argparse
from data.dataset import PascalVOCDataset, NUM_CLASSES
from models.model import SegNet
from main import get_model_params, get_training_params, get_data_params
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

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


def train(model, train_dataloader, criterion, optimizer, scheduler, checkpoint_file):
    is_better = True
    prev_loss = float('inf')
    best_loss = float('inf')

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

        scheduler.step()
        delta = time.time() - start_time
        # is_better = curr_loss < prev_loss
        is_best = curr_loss < best_loss

        # if is_better:
        #     prev_loss = curr_loss
        #     checkpoint_info = {
        #         'epoch': epoch + 1,
        #         'loss': curr_loss,
        #     }
        #     # Save the model's state dictionary with the key 'model_state_dict'
        #     torch.save({
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'checkpoint_info': checkpoint_info
        #     }, checkpoint_file)

        if is_best:
            best_loss = curr_loss
            checkpoint_info = {
                'epoch': epoch + 1,
                'loss': curr_loss,
            }
            # Save the best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'checkpoint_info': checkpoint_info
            }, checkpoint_file)


        print("Epoch #{}/{}\tLoss: {:.8f}\t Time: {:.2f}s".format(epoch + 1, epochs, curr_loss, delta))


if __name__ == "__main__":
    CUDA = args.gpu is not None
    GPU_ID = args.gpu

    print("Loading PascalVOCDataset...")
    train_dataset = PascalVOCDataset(
        list_file=train_data_file,
        img_dir=img_dir,
        mask_dir=mask_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    print("DataLoader...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

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

    # optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    print("train model...")
    train(model, train_dataloader, criterion, optimizer, scheduler, checkpoint_file)
