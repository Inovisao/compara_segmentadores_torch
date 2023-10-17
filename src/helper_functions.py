import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
from data_hyperparameters import MODEL_HYPERPARAMETERS
from torchvision import transforms
from config import (
    VIS_LABEL_MAP as viz_map
)

plt.style.use('ggplot')

def set_class_values(all_classes, classes_to_train):
    """
    Essa função define o número da classe para uma classe específica
    ex.: carro = 0; arvore = 1 e etc...

    :param all_classes: Lista contendo todas as classes.
    :param classes_to_train: Lista contendo os nomes das classes a serem treinadas.
    """
    class_values = [all_classes.index(cls.lower()) for cls in classes_to_train]
    return class_values

def get_label_mask(mask, class_values, label_colors_list):
    """
    Esta função cria a máscara de rótulo a partir da máscara de segmentação da imagem

    :param mask: NP Array, Máscara de Segmentação.
    :param class_values: Lista contendo o valor da classe, ex.: car=0, bus=1.
    :param label_colors_list: Lista contendo o valor RGB de cada classe.
    """
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value
    label_mask = label_mask.astype(int)
    return label_mask

def draw_translucent_seg_maps(
    data, 
    output, 
    epoch, 
    i, 
    val_seg_dir, 
    label_colors_list,
):
    """
    This function color codes the segmentation maps that is generated while
    validating. THIS IS NOT TO BE CALLED FOR SINGLE IMAGE TESTING
    """
    alpha = 1 # how much transparency
    beta = 0.8 # alpha + beta should be 1
    gamma = 0 # contrast

    seg_map = output[0] # use only one output from the batch
    seg_map = torch.argmax(seg_map.squeeze(), dim=0).detach().cpu().numpy()

    image = data[0]
    image = np.array(image.cpu())
    image = np.transpose(image, (1, 2, 0))
    # unnormalize the image (important step)
    # mean = np.array([0.5, 0.5, 0.5])
    # std = np.array([0.5, 0.5, 0.5])
    # image = std * image + mean
    image = np.array(image, dtype=np.float32)
    image = image * 255

    red_map = np.zeros_like(seg_map).astype(np.uint8)
    green_map = np.zeros_like(seg_map).astype(np.uint8)
    blue_map = np.zeros_like(seg_map).astype(np.uint8)


    for label_num in range(0, len(label_colors_list)):
        index = seg_map == label_num
        red_map[index] = np.array(viz_map)[label_num, 0]
        green_map[index] = np.array(viz_map)[label_num, 1]
        blue_map[index] = np.array(viz_map)[label_num, 2]
        
    rgb = np.stack([red_map, green_map, blue_map], axis=2)
    rgb = np.array(rgb, dtype=np.float32)
    # convert color to BGR format for OpenCV
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # cv2.imshow('rgb', rgb)
    # cv2.waitKey(0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, rgb, beta, gamma, image)
    cv2.imwrite(f"{val_seg_dir}/e{epoch}_b{i}.jpg", image)

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    NoImprovement = 0 #Keep track of how many 'epochs' have passed without improvement
    TOLERANCE = MODEL_HYPERPARAMETERS["TOLERANCE"]
    PATIENCE = MODEL_HYPERPARAMETERS["PATIENCE"]
    def __init__(self, best_valid_loss=float('inf'), ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss, epoch, model, out_dir, name='model'):
        bk = False
        if current_valid_loss < (self.best_valid_loss-self.TOLERANCE):
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))
            self.NoImprovement = 0
        else:
            self.NoImprovement += 1
            print(f"Sem melhora há {self.NoImprovement} épocas")
        if self.NoImprovement > self.PATIENCE:
            print(f"Acabou a paciência com {epoch+1} épocas ")
            bk = True
        
        return bk

class SaveBestModelIOU:
    """
    Class to save the best model while training. If the current epoch's 
    IoU is higher than the previous highest, then save the
    model state.
    """
    def __init__(self, best_iou=float(0)):
        self.best_iou = best_iou
        
    def __call__(self, current_iou, epoch, model, out_dir_checkpoints, name='model'):
        if current_iou > self.best_iou:
            self.best_iou = current_iou
            print(f"\nBest validation IoU: {self.best_iou}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir_checkpoints, 'best_'+name+'.pth'))

def save_model(epochs, model, optimizer, criterion, out_dir_checkpoints, name='model'):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(out_dir_checkpoints, name+'.pth'))

def save_plots(
    train_acc, valid_acc, 
    train_loss, valid_loss, 
    train_miou, valid_miou, 
    out_dir_results
):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir_results, 'accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir_results, 'loss.png'))

    # mIOU plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_miou, color='tab:blue', linestyle='-', 
        label='train mIoU'
    )
    plt.plot(
        valid_miou, color='tab:red', linestyle='-', 
        label='validataion mIoU'
    )
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.savefig(os.path.join(out_dir_results, 'miou.png'))

def get_segment_labels(image, model, device):
    image = image.unsqueeze(0).to(device) # add a batch dimension
    with torch.no_grad():
        outputs = model(image)
    return outputs

def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()

    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(viz_map)):
        index = labels == label_num
        red_map[index] = np.array(viz_map)[label_num, 0]
        green_map[index] = np.array(viz_map)[label_num, 1]
        blue_map[index] = np.array(viz_map)[label_num, 2]
        
    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

def image_overlay(image, segmented_image):
    alpha = 1 # transparency for the original image
    beta = 1.0 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image

def save_validation_images(data, outputs, epoch, i, save_dir, label_colors_list):
    """
    Salva as imagens de validação com as máscaras de segmentação em uma pasta específica.

    Args:
        data (torch.Tensor): Imagens de entrada.
        outputs (torch.Tensor): Saídas do modelo (segmentações).
        epoch (int): Número da época atual.
        i (int): Índice da iteração.
        save_dir (str): Diretório raiz para salvar as imagens.
        label_colors_list (list): Lista de cores para cada classe de segmentação.
    """
    # Crie um diretório para a época atual, se ainda não existir
    epoch_dir = os.path.join(save_dir, "epocas", f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    for batch_idx in range(data.size(0)):
        seg_map = outputs[batch_idx]
        seg_map = torch.argmax(seg_map.squeeze(), dim=0).cpu().numpy()

        image = data[batch_idx].cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)

        colored_seg_map = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)

        for label_num in range(len(label_colors_list)):
            index = seg_map == label_num
            colored_seg_map[index] = label_colors_list[label_num]

        # Combine a imagem original com a máscara colorida
        alpha = 0.5  # Ajuste a transparência conforme necessário
        overlaid_image = cv2.addWeighted(image, 1 - alpha, colored_seg_map, alpha, 0)

        # Salve a imagem
        image_filename = f"predicted_{epoch}_{i}_batch_{batch_idx}.png"
        image_path = os.path.join(epoch_dir, image_filename)
        cv2.imwrite(image_path, overlaid_image)

        print(f"Imagem salva em: {image_path}")