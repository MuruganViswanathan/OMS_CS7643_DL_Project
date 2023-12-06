import os
import yaml



#
# load config from yaml file
#
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



if __name__ == "__main__":

    config_path = 'configs/config_segnet.yaml'

    # load config from yaml file
    config = load_config(config_path)

    # get values
    model_name = config['model']['name']
    save_best = config['model']['save_best']
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    learning_rate = config['training']['learning_rate']
    regularization = config['training']['reg']
    momentum = config['training']['momentum']

    # Print or use the values
    print(f"Model Name: {model_name}")
    print(f"save_best: {save_best}")
    print(f"batch_size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"regularization: {regularization}")
    print(f"momentum: {momentum}")

    # get dataset
    # dataset:
    #     name: VOC2012
    #     path: data / VOC2012
    #     train_path: ImageSets / Segmentation / train.txt
    #     img_dir: JPEGImages
    #     mask_dir: SegmentationClass
    #     checkpoint: checkpoints / segnet.pth
    #     out_dir: tests / predictions
    current_dir = os.getcwd()

    dataset_name = config['dataset']['name']

    path = config['dataset']['path']
    dataset_path = os.path.join(current_dir, path)

    tdata = config['dataset']['train_data']
    train_data = os.path.join(current_dir, tdata)

    idir = config['dataset']['img_dir']
    img_dir = os.path.join(current_dir, idir)

    mdir = config['dataset']['mask_dir']
    mask_dir = os.path.join(current_dir, mdir)

    check = config['dataset']['checkpoint']
    checkpoint_dir = os.path.join(current_dir, check)

    odir = config['dataset']['out_dir']
    out_dir = os.path.join(current_dir, odir)

    print(f"current_dir: {current_dir}")
    print(f"dataset_name: {dataset_name}")
    print(f"dataset_path: {dataset_path}")
    print(f"train_data: {train_data}")
    print(f"img_dir: {img_dir}")
    print(f"mask_dir: {mask_dir}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"out_dir: {out_dir}")



