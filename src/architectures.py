import torch.nn as nn
import torchvision.models.segmentation as segmentation
from config import ALL_CLASSES

def get_architecture(architecture):
    def deeplabv3_resnet101(in_channels, out_classes, pretrained):
        model = segmentation.deeplabv3_resnet101(pretrained=pretrained)
        model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier[4] = nn.Conv2d(256, out_classes, kernel_size=1)
        # model.aux_classifier[4] = nn.Conv2d(256, out_classes, kernel_size=1)
        return model

    def deeplabv3_resnet50(in_channels, out_classes, pretrained):
        model = segmentation.deeplabv3_resnet50(pretrained=pretrained)
        model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier[4] = nn.Conv2d(320, out_classes, kernel_size=1)
        return model

    def fcn_resnet50(in_channels, out_classes, pretrained):
        model = segmentation.fcn_resnet50(pretrained=pretrained)
        model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier[4] = nn.Conv2d(512, out_classes, kernel_size=1)
        return model
    
    return locals()[architecture]

#print(get_architecture("deeplabv3_resnet101"))