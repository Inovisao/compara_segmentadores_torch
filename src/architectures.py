import torch.nn as nn
import torchvision.models.segmentation as segmentation
from config import ALL_CLASSES

def resnet101(num_classes=len(ALL_CLASSES)):
    model = segmentation.deeplabv3_resnet101(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

def mobilenetv2(num_classes=len(ALL_CLASSES)):
    model = segmentation.deeplabv3_mobilenet_v2(pretrained=True)
    model.classifier[4] = nn.Conv2d(320, num_classes, kernel_size=1)
    return model

def fcn_resnet50(num_classes=len(ALL_CLASSES)):
    model = segmentation.fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    return model

def pspnet(num_classes=len(ALL_CLASSES)):
    model = segmentation.pspnet(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    return model