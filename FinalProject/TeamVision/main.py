import os
import yaml


config_file = 'configs/config_segnet.yaml'


#
# load config from yaml file
#
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_model_params(config_path=config_file):
    config = load_config(config_path)
    current_dir = os.getcwd()

    check = config['model']['checkpoint']
    checkpoint_file = os.path.join(current_dir, check)

    odir = config['model']['out_dir']
    out_dir = os.path.join(current_dir, odir)

    return (
        config['model']['name'],
        config['model']['save_best'],
        checkpoint_file,
        out_dir,
    )


def get_training_params(config_path=config_file):
    config = load_config(config_path)
    return (
        config['training']['batch_size'],
        config['training']['epochs'],
        config['training']['learning_rate'],
        config['training']['reg'],
        config['training']['momentum'],
    )


def get_data_params(config_path=config_file):
    config = load_config(config_path)
    current_dir = os.getcwd()

    dataset_name = config['dataset']['name']

    path = config['dataset']['path']
    dataset_path = os.path.join(current_dir, path)

    tfile = config['dataset']['train_data']
    train_data_file = os.path.join(dataset_path, tfile)

    vfile = config['dataset']['validation_data']
    val_data_file = os.path.join(dataset_path, vfile)

    idir = config['dataset']['img_dir']
    img_dir = os.path.join(dataset_path, idir)

    mdir = config['dataset']['mask_dir']
    mask_dir = os.path.join(dataset_path, mdir)

    return (
        dataset_name,
        dataset_path,
        train_data_file,
        val_data_file,
        img_dir,
        mask_dir,
    )


if __name__ == "__main__":

    # model values
    model_name, save_best, checkpoint_dir, out_dir = get_model_params()
    print(f"Model Name: {model_name}")
    print(f"save_best: {save_best}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"out_dir: {out_dir}")

    # Training params
    batch_size,  epochs, learning_rate, regularization, momentum = get_training_params()
    print(f"batch_size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"regularization: {regularization}")
    print(f"momentum: {momentum}")

    # dataset
    dataset_name, dataset_path, train_data_file, val_data_file, img_dir, mask_dir = get_data_params()
    print(f"dataset_name: {dataset_name}")
    print(f"dataset_path: {dataset_path}")
    print(f"train_data_dir: {train_data_file}")
    print(f"val_data_file: {val_data_file}")
    print(f"img_dir: {img_dir}")
    print(f"mask_dir: {mask_dir}")
