import glob
import numpy as np
import torch
import albumentations as A
import cv2
from helper_functions import get_label_mask, set_class_values
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,Subset
from data_hyperparameters import DATA_AUGMENTATION, DATA_HYPERPARAMETERS
from sklearn.model_selection import train_test_split

def get_images(root_path):
    train_images = glob.glob(f"{root_path}/train/imagens/*")
    train_images.sort()
    train_masks = glob.glob(f"{root_path}/masks/*")
    train_masks.sort()
    test_images = glob.glob(f"{root_path}/test/imagens/*")
    test_images.sort()
    test_masks = glob.glob(f"{root_path}/masks/*")
    test_masks.sort()
    
    return train_images, train_masks, test_images, test_masks

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

def test_transforms(img_size):
    test_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=False)
    ])
    return test_image_transform

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
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype('int32')

        # if image.shape[1] != mask.shape[1]:
        #     print()
        #     print(image.shape, mask.shape)
        #     print(self.image_paths[index])
        #     print(self.mask_paths[index])
        
        
        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        #plt.imshow(mask)
        #plt.show()
        # Get colored label mask.
        mask = get_label_mask(mask, self.class_values, self.label_colors_list)
        
        
        #plt.imshow(mask)
        #plt.title(self.image_paths[index][:-4].split("/")[-1])
        #plt.show()
        image = np.transpose(image, (2, 0, 1))
        
        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long) 
       # print(image.shape,mask.shape)
        
        
        return image, mask, self.image_paths[index][:-4].split("/")[-1]

def get_dataset(
    train_image_paths, 
    train_mask_paths,
    test_image_paths,
    test_mask_paths,
    all_classes,
    classes_to_train,
    label_colors_list,
    img_size
):
    train_tfms = train_transforms(img_size)
    test_tfms = test_transforms(img_size)

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    test_dataset = SegmentationDataset(
        test_image_paths,
        test_mask_paths,
        test_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    return train_dataset, test_dataset

def print_data_informations(train_data, val_data, test_data, train_dataloader):
    for X, y, _ in train_dataloader:
        print(f"Images batch size: {X.shape[0]}")
        print(f"Number of channels: {X.shape[1]}")
        print(f"Height: {X.shape[2]}")
        print(f"Width: {X.shape[3]}")
        print(f"Labels batch size: {y.shape[0]}")
        print(f"Label data type: {y.dtype}")
        break
    
    total_images = len(train_data) + len(val_data) + len(test_data)
    print(f"Total number of images: {total_images}")
    print(f"Number of training images: {len(train_data)} ({100 * len(train_data) / total_images:>2f}%)")
    print(f"Number of validation images: {len(val_data)} ({100 * len(val_data) / total_images:>2f}%)")
    print(f"Number of test images: {len(test_data)} ({100 * len(test_data) / total_images:>2f}%)")
    
    labels_map = DATA_HYPERPARAMETERS["CLASSES"]
    print(f"\nClasses: {labels_map}")

def get_data_loaders(train_dataset,test_dataset,val_split=DATA_HYPERPARAMETERS["VAL_SPLIT"], batch_size=DATA_HYPERPARAMETERS["BATCH_SIZE"]):
    train_idx, val_idx = train_test_split(list(range(len(train_dataset))), test_size=val_split)
    
    val_dataset = Subset(train_dataset, val_idx)
    train_dataset = Subset(train_dataset, train_idx)
    
    train_dataloader = DataLoader(
        train_dataset, batch_size, drop_last=False, num_workers=10
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size, drop_last=False, num_workers=10
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size, drop_last=False, num_workers=10
    )

    print_data_informations(train_dataset, val_dataset, test_dataset, train_dataloader)
    
    return train_dataloader,val_dataloader,test_dataloader