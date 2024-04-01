#from config import ALL_CLASSES
import os
import torch
from config import load_class_data

ROOT_DATA_DIR = "../data"
TRAIN_DATA_DIR = os.path.join(ROOT_DATA_DIR, "train")
TEST_DATA_DIR = os.path.join(ROOT_DATA_DIR, "test")

class_data  = load_class_data()
ALL_CLASSES, LABEL_COLORS_LIST = class_data["ALL_CLASSES"], class_data["LABEL_COLORS_LIST"]

DATA_HYPERPARAMETERS = {
    "IN_CHANNELS": 3,
    "IMAGE_SIZE": 512,
    "BATCH_SIZE": 6,
    "VAL_SPLIT" : 0.2,
    "CLASSES" : ALL_CLASSES,
    "LABEL_COLORS_LIST": LABEL_COLORS_LIST,
    "NUM_CLASSES": len(ALL_CLASSES),
    "ROOT_DATA_DIR": ROOT_DATA_DIR,
    "TRAIN_DATA_DIR": TRAIN_DATA_DIR,
    "TEST_DATA_DIR": TEST_DATA_DIR,
    #"USE_DATA_AUGMENTATION": True,
    "APENAS_TESTA" : False
}

MODEL_HYPERPARAMETERS = {
    "EPOCHS" : 1000,
    "USE_TRANSFER_LEARNING" : True,
    "PATIENCE" : 50,
    "TOLERANCE" : 0,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "USE_LR_SCHEDULER" : False, #Learning Rate se ajusta a cada X épocas
    "LR_SCHEDULER" : 50 #epocas para ajuste do scheduler
}

DATA_AUGMENTATION = {
    
    "HORIZONTAL_FLIP": 0.5,
    "BRIGHTNESS_CONTRAST": 0.2,
    "SUN_FLARE": 0.2,
    "RANDOM_FOG": 0.2, #Efeito de neblina
    "ROTATION": 25,
}
