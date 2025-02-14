import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
from data_hyperparameters import MODEL_HYPERPARAMETERS, DATA_HYPERPARAMETERS
from args import get_args
from config import load_class_data
plt.style.use('ggplot')

viz_map = load_class_data()["VIS_LABEL_MAP"]

def set_class_values(all_classes, classes_to_train):
    """
    Essa função define o número da classe para uma classe específica
    ex.: carro = 0; arvore = 1 e etc...

    :param all_classes: Lista contendo todas as classes.
    :param classes_to_train: Lista contendo os nomes das classes a serem treinadas.
    """
    class_values = [all_classes.index(cls) for cls in classes_to_train]
    return class_values

def get_label_mask(mask, class_values, label_colors_list):
    """
    Esta função cria a máscara de rótulo a partir da máscara de segmentação da imagem

    :param mask: NP Array, Máscara de Segmentação.
    :param class_values: Lista contendo o valor da classe, ex.: car=0, bus=1.
    :param label_colors_list: Lista contendo o valor RGB de cada classe.
    """
    
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    
    for class_id, color in zip(class_values, label_colors_list):
        boolean_mask = np.all((mask[:, :, :] == color), axis=-1)
        label_mask[boolean_mask] = class_id
        
    
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
        
    def __call__(self, current_valid_loss, epoch, model, out_dir, name):
        bk = False
        if current_valid_loss < (self.best_valid_loss-self.TOLERANCE):
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, name+'.pth'))
            self.NoImprovement = 0
        else:
            self.NoImprovement += 1
            print(f"Sem melhora há {self.NoImprovement} épocas")
        if self.NoImprovement > self.PATIENCE:
            print(f"Acabou a paciência com {epoch+1} épocas ")
            bk = True
        
        return bk

def save_plots(
    train_acc, valid_acc, 
    train_loss, valid_loss, 
    train_miou, valid_miou, 
    out_dir_results
):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Define the factor for adjusting the y-axis limits (1.5 in this example).
    adjustment_factor = 1.5
    
    # Loss and Accuracy plots.
    for plot_type, data, ylabel in [
        ('accuracy', [train_acc, valid_acc], 'Accuracy'),
        ('loss', [train_loss, valid_loss], 'Loss'),
        ('miou', [train_miou, valid_miou], 'mIoU')
    ]:
        plt.figure(figsize=(10, 7))
        for i, dataset in enumerate(['train', 'validation']):
            plt.plot(
                data[i], color=f'tab:{"blue" if i == 0 else "red"}', linestyle='-',
                label=f'{dataset} {plot_type}'
            )

        # Calculate limits based on quantiles and IQR.
        all_losses = data[0] + data[1]
        quantiles = np.quantile(all_losses, [0.25, 0.75])
        iqr = quantiles[1] - quantiles[0]
        norm_sup = quantiles[1] + (adjustment_factor * iqr)
        #norm_inf = quantiles[0] - (adjustment_factor * iqr)
        
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.legend()
        
        # Set the y-axis limits based on calculated values.
        plt.ylim(0, norm_sup)

        # Save the plot to a file based on the plot_type.
        plt.savefig(os.path.join(out_dir_results, f'{plot_type}.png'))
        plt.close()
        
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
        
        
def plot_segmentation(prediction, filename):
    
    color_map = DATA_HYPERPARAMETERS["LABEL_COLORS_LIST"]
    
    img_size = DATA_HYPERPARAMETERS["IMAGE_SIZE"]
    
    args = get_args()
    fold_name = (f"{args['architecture']}_{args['optimizer']}_{args['learning_rate']}")
    
    if not os.path.exists("../results_dl/masks"):
        os.mkdir("../results_dl/masks")
    
    if not os.path.exists(f"../results_dl/masks/{fold_name}"):
        os.mkdir(f"../results_dl/masks/{fold_name}")
    
    if not os.path.exists("../results_dl/predictions"):
        os.mkdir("../results_dl/predictions")
    
    
    if not os.path.exists(f"../results_dl/predictions/{fold_name}"):
        os.mkdir(f"../results_dl/predictions/{fold_name}")
    
    for l, pred in enumerate(prediction):
        print("Plotando máscara da imagem: ", filename[l])
        
        original_image_path = os.path.join("../data/all/imagens", filename[l]+'.jpg')
        original_image = cv2.imread(original_image_path)
        
        plot = np.zeros(shape=(img_size, img_size, DATA_HYPERPARAMETERS["IN_CHANNELS"]), dtype="uint8")
        for i in range(img_size):
            for j in range(img_size):
                plot[i,j,:] = color_map[pred[i, j].to(int)]
                
        plot = cv2.resize(plot, (original_image.shape[1], original_image.shape[0]))
        
        mask_filename = (f'{args["run"]}_mask_{filename[l]}.png')
        mask_path = os.path.join("../results_dl/masks/", fold_name, mask_filename)
        cv2.imwrite(mask_path, plot)
        print(f"Máscara salva em: {mask_path}")
        
        

    for k, pred in enumerate(prediction):
        print("Plotando teste da imagem:", filename[k])
        
        # Carregue a imagem original
        original_image_path = os.path.join("../data/all/imagens", filename[k]+'.jpg')
        original_image = cv2.imread(original_image_path)
        
        plot = np.zeros(shape=(img_size, img_size, DATA_HYPERPARAMETERS["IN_CHANNELS"]), dtype="uint8")
        for i in range(img_size):
            for j in range(img_size):
                plot[i, j, :] = color_map[pred[i, j].to(int)]
        
        plot = cv2.resize(plot, (original_image.shape[1], original_image.shape[0]))
        
        alpha = 0.5
        overlaid_image = cv2.addWeighted(original_image, 1 - alpha, plot, alpha, 0)
        
        img_filename = f'{args["run"]}_{filename[k]}.png'
        img_path = os.path.join("../results_dl/predictions", fold_name, img_filename)
        cv2.imwrite(img_path, overlaid_image)
        print(f"Imagem salva em: {img_path}")