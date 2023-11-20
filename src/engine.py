#Código baseado no tutorial do Sovit Ranjan Rath com algumas alterações (https://debuggercafe.com/multi-class-semantic-segmentation-training-using-pytorch/#download-code)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import metrics
import torch
from data_hyperparameters import MODEL_HYPERPARAMETERS, DATA_HYPERPARAMETERS
from helper_functions import draw_translucent_seg_maps, plot_segmentation
from metrics import IOUEval
import sys
import numpy as np

device = MODEL_HYPERPARAMETERS["DEVICE"]
#test
def train(
    model,
    train_dataloader,
    device,
    optimizer,
    criterion,
    classes_to_train
):
    print('Training')
    model.train()
    train_running_loss = 0.0
    counter = 0 # to keep track of batch counter
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)

    for i, data in enumerate(train_dataloader):
        counter += 1
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        if data.shape[0] == 1:
            continue
        outputs = model(data)['out']
        
        ##### BATCH-WISE LOSS #####
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        ###########################
        for image_idx in range(len(data)):
            counter += 1
            # Calcula o índice do lote para a imagem atual
            batch_index = i * len(data) + image_idx

            # Imprima informações a cada imagem processada
            if batch_index % 10 == 0:
                loss_show = loss.item()
                print(f"Loss: {loss_show:.7f}")
                sys.stdout.flush()
        ##### BACKPROPAGATION AND PARAMETER UPDATION #####
        loss.backward()
        optimizer.step()
        ##################################################

        iou_eval.addBatch(outputs.max(1)[1].data, target.data)
        
    ##### PER EPOCH LOSS #####
    train_loss = train_running_loss / counter

    ##########################
    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval.getMetric()
    return train_loss, overall_acc, mIOU

def validate(
    model,
    valid_dataloader,
    device,
    criterion,
    classes_to_train,
    label_colors_list,
    epoch,
    save_dir
):
    print('Validating')
    model.eval()
    valid_running_loss = 0
    # Calculate the number of batches.
    num_batches = len(valid_dataloader)
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)

    with torch.no_grad():

        counter = 0 # To keep track of batch counter.
        for i, data in enumerate(valid_dataloader):
            counter += 1
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)['out']
            
            # Save the validation segmentation maps every
            # last batch of each epoch
            if i == num_batches - 1:
                draw_translucent_seg_maps(
                    data, 
                    outputs, 
                    epoch, 
                    i, 
                    save_dir, 
                    label_colors_list,
                )

            ##### BATCH-WISE LOSS #####
            loss = criterion(outputs, target)
            valid_running_loss += loss.item()
            ###########################

            iou_eval.addBatch(outputs.max(1)[1].data, target.data)
        
        ##### PER EPOCH LOSS #####
    valid_loss = valid_running_loss / counter
    ##########################
    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval.getMetric()
    return valid_loss, overall_acc, mIOU

def test(dataloader, model, path_to_save_matrix_csv, path_to_save_matrix_png, labels_map):
    """
    This function tests a model.
    Args:
        dataloader: the test dataloader.
        model: the model to be tested.
        path_to_save_matrix_csv: the path to save the confusion matrix as a .csv file.
        path_to_save_matrix_png: the path to save the confusion matrix as a .png image.
        labels_map: a list with the labels. It will be used to create a list with the wrong classification.

    Returns: precision, recall and fscore calculated for the model in regard to the predictions on the test dataset.

    """
    # Put the model in evaluation mode.
    model.eval()

    # Get the total number of images.
    num_images = len(dataloader.dataset)

    # Initialize empty lists for predictions and labels.
    predictions, labels = [], []

    # Initialize the number of correct predictions with value 0.
    test_correct = 0
    
    plot = plt.figure(num=1)
    
    #Proceed without calculating the gradients.
    with torch.no_grad():
        # Iterate over the data.
        for img, label, filename in dataloader:
            # Send images and labels to the correct device.
            img, label = img.to(device, dtype=torch.float), label.to(device)

            # Make predictions with the model.
            prediction = dict(model(img))["out"]
        
            #prediction_prob_values = softmax(prediction)
    
            # Get the index of the prediction with the highest probability.
            prediction = prediction.argmax(1)
            
            plot_segmentation(prediction.cpu(), filename, plot)
            
            # Append predictions and labels to the lists initialized earlier.
            # Also, send both predictions and labels to the cpu.
            predictions.extend(prediction.cpu().flatten().tolist())
            labels.extend(label.cpu().flatten().tolist())

            # Accumulate the number of correct predictions.
            test_correct += (prediction == label).type(torch.float).sum().item()
            

    # Calculate pixels value.
    pixels = num_images*(DATA_HYPERPARAMETERS["IMAGE_SIZE"]**2)
    
    # Calculate the accuracy.
    acc = test_correct / pixels

    # Get the classes for the matrix.
    classes = DATA_HYPERPARAMETERS["CLASSES"]
    
    # Create Jaccard similarity coefficient score
    miou = metrics.jaccard_score(y_true=labels, y_pred=predictions, labels=[i for i in range(len(classes))])
    
    # Create the confusion matrix.
    matrix = metrics.confusion_matrix(y_true=labels, y_pred=predictions, labels=[i for i in range(len(classes))])
    
    # Convert the matrix into a pandas dataframe.
    df_matrix = pd.DataFrame(matrix)
    
    
    # Save the matrix as a csv file.
    df_matrix.to_csv(path_to_save_matrix_csv)
    
    # Create a graphical matrix.
    plt.figure()
    sn.heatmap(df_matrix,annot=True,xticklabels=classes,yticklabels=classes,fmt='d')
    plt.title("Pixels matrix", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.grid(False)
    
    # Save the figure.
    plt.savefig(path_to_save_matrix_png, bbox_inches="tight")
    
    # Get some metrics.
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(labels, predictions,average='macro',zero_division=0)

    # Get some classes metrics
    class_precision, class_recall, class_fscore, _ = metrics.precision_recall_fscore_support(labels, predictions, zero_division=0)
    
    # Write some results.
    print(f"Total number of predictions: {len(dataloader.dataset)}.")
    print(f"Number of correct predictions: {test_correct}.")
    print(f"Test accuracy: {(100 * acc):>0.2f}%.\n")
    print('\nPerformance metrics in the test set:')
    print(metrics.classification_report(labels, predictions, target_names=classes))

    # Return the metrics.
    return precision, recall, fscore, miou, class_precision, class_recall, class_fscore