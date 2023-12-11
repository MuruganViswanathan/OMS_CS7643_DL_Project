import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

plt.switch_backend('agg')
plt.axis('off')


def visualize_predictions(input_image, predicted_mask, target_mask, out_dir, idx, batch_idx):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(input_image.permute(1, 2, 0))
    axes[0].set_title('Input Image')

    axes[1].imshow(predicted_mask, cmap='jet')
    axes[1].set_title('Predicted Mask')

    axes[2].imshow(target_mask, cmap='jet')
    axes[2].set_title('Ground Truth')

    save_path = os.path.join(out_dir, f"prediction_{batch_idx}_{idx}.png")
    plt.savefig(save_path)
    plt.close(fig)
    return save_path
