import glob
import numpy as np
import os
import torch
import albumentations as A
import cv2

from helper_functions import get_label_mask, set_class_values
from torch.utils.data import Dataset, DataLoader
from data_hyperparameters import DATA_AUGMENTATION, DATA_HYPERPARAMETERS

def get_images(root_path):
    print(root_path)
    train_images = glob.glob(f"{root_path}/train/imagens/*")
    train_images.sort()
    train_masks = glob.glob(f"{root_path}/masks/*")
    train_masks.sort()
    valid_images = glob.glob(f"{root_path}/test/imagens/*")
    valid_images.sort()
    valid_masks = glob.glob(f"{root_path}/masks/*")
    valid_masks.sort()

    return train_images, train_masks, valid_images, valid_masks

def train_transforms(img_size):
    """
    Transforms/augmentations for training images and masks.

    :param IMAGE_SIZE: Integer, for image resize.
    """
    train_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.HorizontalFlip(DATA_AUGMENTATION["HORIZONTAL_FLIP"]),
        A.RandomBrightnessContrast(DATA_AUGMENTATION["BRIGHTNESS_CONTRAST"]),
        #A.RandomSunFlare(DATA_AUGMENTATION["SUN_FLARE"]),
        A.RandomFog(DATA_AUGMENTATION["RANDOM_FOG"]),
        A.Rotate(DATA_AUGMENTATION["ROTATION"]),
    ])
    return train_image_transform

def valid_transforms(img_size):
    """
    Transforms/augmentations for validation images and masks.

    :param img_size: Integer, for image resize.
    """
    valid_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
    ])
    return valid_image_transform

class SegmentationDataset(Dataset):
    def __init__(
        self, 
        image_paths, 
        mask_paths, 
        tfms, 
        label_colors_list,
        classes_to_train,
        all_classes
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        # Convert string names to class values for masks.
        self.class_values = set_class_values(
            self.all_classes, self.classes_to_train
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        image = image / 255.0
        mask_name = str("../data/masks/" + self.image_paths[index][:-4].split("/")[-1] + ".png")
        mask = cv2.imread(mask_name, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype('float32')
        
        if image.shape[1] != mask.shape[1]:
            print()
            print(image.shape, mask.shape)
            print(self.image_paths[index])
            print(self.mask_paths[index])
        
        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Get colored label mask.
        mask = get_label_mask(mask, self.class_values, self.label_colors_list)
       
        image = np.transpose(image, (2, 0, 1))
        
        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long) 

        return image, mask

def get_dataset(
    train_image_paths, 
    train_mask_paths,
    valid_image_paths,
    valid_mask_paths,
    all_classes,
    classes_to_train,
    label_colors_list,
    img_size
):
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    return train_dataset, valid_dataset

def get_data_loaders(train_dataset, valid_dataset, batch_size):
    train_data_loader = DataLoader(
        train_dataset, batch_size=DATA_HYPERPARAMETERS["BATCH_SIZE"], drop_last=False, num_workers=10
    )
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=DATA_HYPERPARAMETERS["BATCH_SIZE"], drop_last=False, num_workers=10
    )

    return train_data_loader, valid_data_loader