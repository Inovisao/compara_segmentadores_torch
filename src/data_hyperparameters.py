from config import ALL_CLASSES
import os

ROOT_DATA_DIR = "../data"
TRAIN_DATA_DIR = os.path.join(ROOT_DATA_DIR, "train")
TEST_DATA_DIR = os.path.join(ROOT_DATA_DIR, "test")


DATA_HYPERPARAMETERS = {
    "IN_CHANNELS": 3,
    "IMAGE_SIZE": 512,
    "BATCH_SIZE": 4,
    "NUM_CLASSES": len(ALL_CLASSES),
    "ROOT_DATA_DIR": ROOT_DATA_DIR,
    "TRAIN_DATA_DIR": TRAIN_DATA_DIR,
    "TEST_DATA_DIR": TEST_DATA_DIR,
    #"USE_DATA_AUGMENTATION": True,
}

MODEL_HYPERPARAMETERS = {
    "EPOCHS" : 100,
    "USE_TRANSFER_LEARNING" : True,
    "PATIENCE" : 10,
    "TOLERANCE" : 0.1,
    "USE_LR_SCHEDULER" : False, #Learning Rate se ajusta a cada X épocas
    "LR_SCHEDULER" : 60 #epocas para ajuste do scheduler
}

DATA_AUGMENTATION = {
    
    "HORIZONTAL_FLIP": 0.5,
    "BRIGHTNESS_CONTRAST": 0.2,
    "SUN_FLARE": 0.2,
    "RANDOM_FOG": 0.2, #Efeito de neblina
    "ROTATION": 25,
}
