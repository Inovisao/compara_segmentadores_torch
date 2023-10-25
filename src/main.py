import torch
import torch.nn as nn
import os
import argparse
from data_manager import get_images, get_dataset, get_data_loaders
from engine import train, validate, test
from config import ALL_CLASSES, LABEL_COLORS_LIST
from data_hyperparameters import DATA_HYPERPARAMETERS, MODEL_HYPERPARAMETERS, DATA_AUGMENTATION
from arch_optim import get_optimizer,get_architecture
from helper_functions import SaveBestModel, save_plots
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
    arg_parser.add_argument("-o", "--optimizer", required=True, default='adam', type=str)

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
    out_dir_checkpoints = os.path.join('..','model_checkpoints')
    out_dir_results = os.path.join('..','results','history')
    #out_dir_plots = os.path.join('..','outputs','results')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_valid_preds, exist_ok=True)
    os.makedirs(out_dir_checkpoints, exist_ok=True)
   # os.makedirs(out_dir_plots,exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_architecture(args ["architecture"],
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

    optimizer = get_optimizer(optimizer=args["optimizer"], model=model, learning_rate=args['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    train_images, train_masks, test_images, test_masks = get_images(
        root_path='../data'
    )



    classes_to_train = ALL_CLASSES

    train_dataset, test_dataset = get_dataset(
        train_images, 
        train_masks,
        test_images,
        test_masks,
        ALL_CLASSES,
        classes_to_train,
        LABEL_COLORS_LIST,
        img_size=DATA_HYPERPARAMETERS["IMAGE_SIZE"],
    )

    train_dataloader,val_dataloader, test_dataloader = get_data_loaders(train_dataset, test_dataset, batch_size=DATA_HYPERPARAMETERS['BATCH_SIZE'])

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()

    # LR Scheduler.
    scheduler = MultiStepLR(
        optimizer, milestones=[MODEL_HYPERPARAMETERS["LR_SCHEDULER"]], gamma=0.1, verbose=False
        )
    
    name = str(f"{args['run']}_{args['architecture']}_{args['optimizer']}")

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
        valid_epoch_loss, valid_epoch_pixacc, valid_epoch_miou= validate(
            model,
            val_dataloader,
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

    
        patience_is_over = save_best_model(valid_epoch_loss, 
                           epoch, 
                           model, 
                           out_dir_checkpoints, 
                           name)
        
        if patience_is_over: break
        
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
        if MODEL_HYPERPARAMETERS["USE_LR_SCHEDULER"] == True:
            scheduler.step()
        else:
            pass
        print('-' * 50)


 
    # Save the loss and accuracy plots.
    save_plots(
        train_pix_acc, valid_pix_acc, 
        train_loss, valid_loss,
        train_miou, valid_miou,
        out_dir_results,
    )
    print('TRAINING COMPLETE')

    #Carrega o modelo pré-treinado
    model.load_state_dict(torch.load(os.path.join(out_dir_checkpoints, name + ".pth"))["model_state_dict"])
    
    # Define the paths to save the confusion matrix files.
    if not os.path.exists("../results/matrix/"):
        os.makedirs("../results/matrix/")
        
    path_to_matrix_csv = "../results/matrix/" + name + "_MATRIX.csv"    
    path_to_matrix_png = "../results/matrix/" + name + "_MATRIX.png"
    

    # Test, save the results and get precision, recall and fscore.
    precision, recall, fscore = test(dataloader=test_dataloader,
                                                      model=model, 
                                                      path_to_save_matrix_csv=path_to_matrix_csv, 
                                                      path_to_save_matrix_png=path_to_matrix_png,
                                                      labels_map=DATA_HYPERPARAMETERS["CLASSES"])


    #Create a string with run, learning rate, architecture,
    # optimizer, precision, recall and fscore, to append to the csv file:
    results = str(args["run"]) + "," + str(args["learning_rate"]) + "," + str(args["architecture"]) + \
        "," + str(args["optimizer"]) + "," + str(precision) + "," + str(recall) + "," + str(fscore) + "\n"

    # Open file, write and close.
    f = open("../results_dl/results.csv", "a")
    f.write(results)
    f.close()
