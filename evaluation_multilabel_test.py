import csv
import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
from torch.optim import Adagrad, Adam, SGD
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import src.architectures as architectures
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from src.segformer import make_SegFormerB1


class COCOMultiLabelSegmentationDataset(Dataset):
    def __init__(self, root_dir, annotation_file, include_patterns=None, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)

        # Filter image IDs based on the include_patterns
        self.image_ids = []
        for image_id in self.coco.imgs.keys():
            img_info = self.coco.loadImgs(image_id)[0]
            img_path = os.path.join(self.root_dir, img_info['file_name'])
            if os.path.exists(img_path):
                if include_patterns:
                    if any(pattern in img_info['file_name'] for pattern in include_patterns):
                        self.image_ids.append(image_id)
                else:
                    self.image_ids.append(image_id)

        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # Create an empty mask
        mask = np.zeros(
            (img_info['height'], img_info['width']), dtype=np.uint8)

        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann)
                              * (ann['category_id'] + 1))

        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def load_model_and_optimizer(model_name, optimizer_name, checkpoint_path):

    def deeplabv3_resnet101(in_channels, out_classes, pretrained):
        model = segmentation.deeplabv3_resnet101(pretrained=pretrained)
        model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier[4] = nn.Conv2d(256, out_classes, kernel_size=1)
        return model

    def deeplabv3_resnet50(in_channels, out_classes, pretrained):
        model = segmentation.deeplabv3_resnet50(pretrained=pretrained)
        model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier[4] = nn.Conv2d(320, out_classes, kernel_size=1)
        return model

    def fcn_resnet50(in_channels, out_classes, pretrained):
        model = segmentation.fcn_resnet50(pretrained=pretrained)
        model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier[4] = nn.Conv2d(512, out_classes, kernel_size=1)
        return model

    def segformer(in_channels, out_classes, pretrained):
        # , pretrained=pretrained)
        model = make_SegFormerB1(
            num_classes=out_classes, channels=in_channels, pretrained=pretrained)
        if not pretrained:
            # Lógica para baixar os pesos e carregar no modelo.
            pretrained = "checkpoint/segformer.b1.512x512.ade.160k.pth"

            pass
        # model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # model.classifier[4] = nn.Conv2d(512, out_classes, kernel_size=1)
        return model
    # Define model architectures
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device('cuda:0'))

    # Determine the number of classes from the checkpoint
    print(checkpoint['model_state_dict']['classifier.4.weight'].shape[0])
    num_classes = checkpoint['model_state_dict']['classifier.4.weight'].shape[0]

    # Model setup
    model = None
    if model_name == 'deeplabv3_resnet101':
        model = deeplabv3_resnet101(
            3, num_classes, True)
    elif model_name == 'deeplabv3_resnet50':
        model = deeplabv3_resnet50(3, num_classes, True)
    elif model_name == 'fcn_resnet50':
        model = fcn_resnet50(3, num_classes, True)
    elif model_name == 'segformer':
        model = segformer(3, num_classes, True)
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    model.load_state_dict(checkpoint['model_state_dict'])

    # Optimizer setup
    optimizer = None
    if optimizer_name == 'adagrad':
        optimizer = Adagrad(model.parameters())
    elif optimizer_name == 'adam':
        optimizer = Adam(model.parameters())
    elif optimizer_name == 'sgd':
        optimizer = SGD(model.parameters())
    else:
        raise ValueError(f'Unknown optimizer name: {optimizer_name}')

    return model, optimizer
# Assuming the evaluate function from before


def evaluate(model, dataloader, device, num_classes):
    all_preds = []
    all_labels = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            # Remove the extra dimension in labels if it exists
            labels = labels.squeeze(1)
            outputs = model(images)['out']
            # Convert to discrete class labels
            preds = torch.argmax(outputs, dim=1)

            # Move to CPU and convert to numpy arrays
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return all_preds, all_labels

# Function to calculate metrics


def calculate_metrics(preds, labels, num_classes):
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    valid_mask = (labels_flat >= 0) & (labels_flat < num_classes)
    preds_flat = preds_flat[valid_mask]
    labels_flat = labels_flat[valid_mask]

    preds_flat = preds_flat.astype(int)
    labels_flat = labels_flat.astype(int)

    precision = precision_score(
        labels_flat, preds_flat, average='macro', labels=np.arange(num_classes))
    recall = recall_score(labels_flat, preds_flat,
                          average='macro', labels=np.arange(num_classes))
    f1 = f1_score(labels_flat, preds_flat, average='macro',
                  labels=np.arange(num_classes))

    class_precision = precision_score(
        labels_flat, preds_flat, average=None, labels=np.arange(num_classes))
    class_recall = recall_score(
        labels_flat, preds_flat, average=None, labels=np.arange(num_classes))
    class_f1 = f1_score(labels_flat, preds_flat,
                        average=None, labels=np.arange(num_classes))

    def mean_iou(pred, true, num_classes):
        ious = []
        for cls in range(num_classes):
            pred_cls = pred == cls
            true_cls = true == cls
            intersection = np.logical_and(pred_cls, true_cls).sum()
            union = np.logical_or(pred_cls, true_cls).sum()
            if union == 0:
                ious.append(float('nan'))
            else:
                iou = intersection / union
                ious.append(iou)
        return np.nanmean(ious)

    miou = mean_iou(preds, labels, num_classes)

    return precision, recall, f1, class_precision, class_recall, class_f1, miou

# Function to visualize predictions


def save_predictions(data_loader, model, device, save_dir, num_images=5):
    model.eval()
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        preds = model(images)
        preds = torch.argmax(preds, dim=1)

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(min(num_images, images.size(0))):
        # Save input image
        plt.figure()
        plt.imshow(images[i].cpu().permute(1, 2, 0))
        plt.title('Input Image')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'input_{i}.png'))
        plt.close()

        # Save ground truth image
        plt.figure()
        plt.imshow(labels[i].cpu(), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'ground_truth_{i}.png'))
        plt.close()

        # Save predicted image
        plt.figure()
        plt.imshow(preds[i].cpu(), cmap='gray')
        plt.title('Predicted Label')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'predicted_{i}.png'))
        plt.close()


def visualize_predictions(images, labels, preds, num_images=5):
    for i in range(min(num_images, len(images))):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(labels[i].cpu().numpy(), cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(preds[i].cpu().numpy(), cmap='gray')
        plt.show()


# Define any transformations (e.g., resizing, normalization) to apply to the images and masks
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Create an instance of the dataset
root_dir = '/home/corbusier/development/compara_segmentadores_torch_/data_512_UCDB_UPS/all/imagens'
annotation_file = '/home/corbusier/development/compara_segmentadores_torch_/data_512_UCDB_UPS/annotations_coco_json/_annotations.coco.json'
save_directory = '/home/corbusier/development/compara_segmentadores_torch_/Evaluation_from_checkpoints/predictions'
# include_patterns = ['UPS', 'UCDB']
include_patterns = ['UPS']

dataset = COCOMultiLabelSegmentationDataset(
    root_dir=root_dir, annotation_file=annotation_file, include_patterns=include_patterns, transform=image_transform)

# Create a DataLoader for your dataset
val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

# Example usage: iterate over the DataLoader
for images, masks in val_loader:
    # Perform your evaluation here
    pass

# Define the base path where the checkpoints are stored
base_path = '/home/corbusier/development/compara_segmentadores_torch_/Evaluation_from_checkpoints/checkpoints/model_checkpoints/'

# List of checkpoint files
checkpoint_files = [
    '1_deeplabv3_resnet101_adagrad.pth',
    '1_deeplabv3_resnet101_adam.pth',
    '1_deeplabv3_resnet101_sgd.pth',
    '1_fcn_resnet50_adagrad.pth',
    '1_fcn_resnet50_adam.pth',
    '1_fcn_resnet50_sgd.pth',
    # '1_segformer_adagrad.pth',
    # '1_segformer_adam.pth',
    # '1_segformer_sgd.pth',
    '2_deeplabv3_resnet101_adagrad.pth',
    '2_deeplabv3_resnet101_adam.pth',
    '2_deeplabv3_resnet101_sgd.pth',
    '2_fcn_resnet50_adagrad.pth',
    '2_fcn_resnet50_adam.pth',
    '2_fcn_resnet50_sgd.pth',
    # '2_segformer_adagrad.pth',
    # '2_segformer_adam.pth',
    # '2_segformer_sgd.pth'
]

# Generate the checkpoint dictionary
checkpoints = []

for file in checkpoint_files:
    parts = file.split('_')
    print(parts)
    model_name = f"{parts[1]}_{parts[2]}"
    process = parts[0]
    optimizer_name = parts[3].split('.')[0]
    checkpoint_path = base_path + file
    checkpoints.append({
        'process': process,
        'model_name': model_name,
        'optimizer_name': optimizer_name,
        'checkpoint_path': checkpoint_path
    })

# Print the generated checkpoint dictionary
for checkpoint in checkpoints:
    print(checkpoint)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 3

csv_file = 'evaluation_results.csv'
csv_columns = ['Model', 'Optimizer', 'Precision', 'Recall', 'F1-score',
               'Class Precision', 'Class Recall', 'Class F1-score', 'mIoU']


with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_columns)

    for checkpoint_info in checkpoints:
        model_name = checkpoint_info['model_name']
        optimizer_name = checkpoint_info['optimizer_name']
        checkpoint_path = checkpoint_info['checkpoint_path']

        model, optimizer = load_model_and_optimizer(
            model_name, optimizer_name, checkpoint_path)
        save_predictions(val_loader, model, device, save_directory)
        preds, labels = evaluate(model, val_loader, device, num_classes)
        precision, recall, f1, class_precision, class_recall, class_f1, miou = calculate_metrics(
            preds, labels, num_classes)

        # Write to CSV
        writer.writerow([
            model_name, optimizer_name, precision, recall, f1,
            class_precision.tolist(), class_recall.tolist(), class_f1.tolist(), miou
        ])

        # Print results
        print(f'Results for {model_name} with {optimizer_name}:')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-score: {f1}')
        print(f'Class Precision: {class_precision}')
        print(f'Class Recall: {class_recall}')
        print(f'Class F1-score: {class_f1}')
        print(f'mIoU: {miou}')
        print('--------------------------------------')

        # Visualize some predictions
        images, labels_batch = next(iter(val_loader))
        images, labels_batch = images.to(device), labels_batch.to(device)
        outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1)
        visualize_predictions(images, labels_batch.squeeze(1), preds)
