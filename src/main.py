import torch
import torch.nn as nn
import os
import argparse
from data_manager import get_images, get_dataset, get_data_loaders
from engine import train, validate
from config import ALL_CLASSES, LABEL_COLORS_LIST
from data_hyperparameters import DATA_HYPERPARAMETERS, MODEL_HYPERPARAMETERS, DATA_AUGMENTATION
from arch_optim import get_optimizer,get_architecture
from helper_functions import save_model, SaveBestModel, save_plots, SaveBestModelIOU
from torch.optim.lr_scheduler import MultiStepLR
from architectures import *


def get_args():
    """
    This function gets the arguments of the program, that is, the architecture, the optimizer and the learning rate.

    Returns: a dictionary with the values of the arguments, in which the keys are the names defined for each argument in the second argument of each of the functions below.
    """

    # Instantiate the argument parser.
    arg_parser = argparse.ArgumentParser()

    # Parse the architecture.
    arg_parser.add_argument("-a", "--architecture", required=True, default='deeplabv3_resnet101', type=str)
    
    # Parse the optimizer.
    arg_parser.add_argument("-o", "--optimizer", required=True, default='SGD', type=str)

    # Parse the number of the run.
    arg_parser.add_argument("-r", "--run", required=True, default=1, type=int)
    
    # Parse the learning rate.
    arg_parser.add_argument("-l", "--learning_rate", required=True, default=0.001, type=float)

    # Parse the arguments and return them as a dictionary.
    return vars(arg_parser.parse_args())



if __name__ == '__main__':
    args = get_args()
    # Create a directory with the model name for outputs.
    out_dir = os.path.join('..', 'outputs')
    out_dir_valid_preds = os.path.join('..', 'outputs', 'valid_preds')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_valid_preds, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_architecture(args["architecture"], 
                             in_channels=DATA_HYPERPARAMETERS["IN_CHANNELS"], 
                             out_classes=DATA_HYPERPARAMETERS["NUM_CLASSES"], 
                             pretrained=MODEL_HYPERPARAMETERS["USE_TRANSFER_LEARNING"])

    
    model = model.to(device)


    print("===================================")
    print("==> MODEL")
    print(model)
    print("===================================")
    print("==> MODEL HYPERPARAMETERS")
    print(MODEL_HYPERPARAMETERS)
    print("===================================")
    print("==> DATA HYPERPARAMETERS")
    print(DATA_HYPERPARAMETERS)
    print("===================================")
    print("==> DATA AUGMENTATION")
    print(DATA_AUGMENTATION)
    print("===================================")

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    #optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'])
    optimizer = get_optimizer(optimizer=args["optimizer"], model=model, learning_rate=args['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    train_images, train_masks, valid_images, valid_masks = get_images(
        root_path='../data/'    
    )

    classes_to_train = ALL_CLASSES

    train_dataset, valid_dataset = get_dataset(
        train_images, 
        train_masks,
        valid_images,
        valid_masks,
        ALL_CLASSES,
        classes_to_train,
        LABEL_COLORS_LIST,
        img_size=DATA_HYPERPARAMETERS["IMAGE_SIZE"],
    )

    train_dataloader, valid_dataloader = get_data_loaders(
        train_dataset, valid_dataset, batch_size=DATA_HYPERPARAMETERS['BATCH_SIZE']
    )

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()
    save_best_iou = SaveBestModelIOU()

    # LR Scheduler.
    scheduler = MultiStepLR(
        optimizer, milestones=[60], gamma=0.1, verbose=True
    )

    EPOCHS = MODEL_HYPERPARAMETERS["EPOCHS"]
    train_loss, train_pix_acc, train_miou = [], [], []
    valid_loss, valid_pix_acc, valid_miou = [], [], []
    for epoch in range (EPOCHS):
        print(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_epoch_pixacc, train_epoch_miou = train(
            model,
            train_dataloader,
            device,
            optimizer,
            criterion,
            classes_to_train
        )
        valid_epoch_loss, valid_epoch_pixacc, valid_epoch_miou = validate(
            model,
            valid_dataloader,
            device,
            criterion,
            classes_to_train,
            LABEL_COLORS_LIST,
            epoch,
            save_dir=out_dir_valid_preds
        )
        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pixacc)
        train_miou.append(train_epoch_miou)
        valid_loss.append(valid_epoch_loss)
        valid_pix_acc.append(valid_epoch_pixacc)
        valid_miou.append(valid_epoch_miou)

        save_best_model(
            valid_epoch_loss, epoch, model, out_dir, name='model_loss'
        )
        save_best_iou(
            valid_epoch_miou, epoch, model, out_dir, name='model_iou'
        )

        print(
            f"Train Epoch Loss: {train_epoch_loss:.4f},",
            f"Train Epoch PixAcc: {train_epoch_pixacc:.4f},",
            f"Train Epoch mIOU: {train_epoch_miou:4f}"
        )
        print(
            f"Valid Epoch Loss: {valid_epoch_loss:.4f},", 
            f"Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}",
            f"Valid Epoch mIOU: {valid_epoch_miou:4f}"
        )
        if scheduler:
            scheduler.step()
        print('-' * 50)

    save_model(EPOCHS, model, optimizer, criterion, out_dir, name='model')
    # Save the loss and accuracy plots.
    save_plots(
        train_pix_acc, valid_pix_acc, 
        train_loss, valid_loss,
        train_miou, valid_miou, 
        out_dir
    )
    print('TRAINING COMPLETE')