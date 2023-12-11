import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from data.dataset import PascalVOCDataset, NUM_CLASSES
from models.model import SegNet
from main import get_model_params, get_training_params, get_data_params
from utils.visualization import visualize_predictions
import torch

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
parser.add_argument('--model_path')
parser.add_argument('--gpu', type=int)
args = parser.parse_args()

# # Function to visualize predictions
# def visualize_predictions(input_image, predicted_mask, target_mask, out_dir, idx, batch_idx):
#
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
#     plt.savefig(os.path.join(out_dir, f"prediction_{batch_idx}_{idx}.png"))
#     plt.close(fig)


def validate(model, val_dataloader, criterion, out_dir):

    # print("Set to eval mode...")
    model.eval()

    for batch_idx, batch in enumerate(val_dataloader):
        input_tensor = batch['image']
        target_tensor = batch['mask']

        if CUDA:
            input_tensor = input_tensor.cuda(GPU_ID)
            target_tensor = target_tensor.cuda(GPU_ID)

        with torch.no_grad():
            predicted_tensor, softmaxed_tensor = model(input_tensor)
            loss = criterion(predicted_tensor, target_tensor)

        for idx, (predicted_mask, target_mask, input_image) in enumerate(zip(softmaxed_tensor, target_tensor, input_tensor)):
            predicted_mask_np = predicted_mask.argmax(dim=0).cpu().numpy()
            target_mask_np = target_mask.cpu().numpy()

            #visualize_predictions(input_image, predicted_mask_np, target_mask_np, out_dir, idx, batch_idx)
            save_path = visualize_predictions(input_image, predicted_mask_np, target_mask_np, out_dir, idx, batch_idx)
            # print(f"Visualization saved at: {save_path}")


def main():

    if args.model_path is not None and os.path.exists(args.model_path):
        print("Using arg specified model:", args.model_path)
        SAVED_MODEL_PATH = args.model_path
    else:
        print("Using checkpoint_file:", checkpoint_file)
        SAVED_MODEL_PATH = checkpoint_file

    # CUDA = args.gpu is not None
    # GPU_ID = args.gpu

    print("Loading validation PascalVOCDataset...")
    val_dataset = PascalVOCDataset(list_file=val_data_file, img_dir=img_dir, mask_dir=mask_dir)

    print("Validation DataLoader...")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print("prepare SegNet model ...")
    model = SegNet(input_channels=NUM_INPUT_CHANNELS, output_channels=NUM_OUTPUT_CHANNELS)
    if CUDA:
        print("CUDA true, move model to GPU", GPU_ID)
        model = model.cuda(GPU_ID)

    class_weights = 1.0 / val_dataset.get_class_probability()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print("Loading", SAVED_MODEL_PATH)
    #model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    checkpoint = torch.load(SAVED_MODEL_PATH)

    # Check if 'model_state_dict' key is present, otherwise assume direct state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    out_dir = os.path.join(os.path.dirname(SAVED_MODEL_PATH), 'predictions')
    os.makedirs(out_dir, exist_ok=True)

    print("Validate...")
    validate(model, val_dataloader, criterion, out_dir)


if __name__ == "__main__":
    CUDA = args.gpu is not None
    GPU_ID = args.gpu
    main()
