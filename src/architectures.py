import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
from torchvision import models
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
from segformer import make_SegFormerB1
from data_hyperparameters import DATA_HYPERPARAMETERS
import os

pasta_checkpoints=os.path.join(os.getcwd(),'checkpoints')
print('Pasta com os checkpoints (*.pth): ',pasta_checkpoints)

def architectures():
        def deeplabv3_resnet101(in_channels, out_classes, pretrained):
            model = segmentation.deeplabv3_resnet101(pretrained=pretrained)
            model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.classifier[4] = nn.Conv2d(256, out_classes, kernel_size=1)
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
        
        def segformer(in_channels, out_classes, pretrained):
            model = make_SegFormerB1(num_classes=out_classes, channels=in_channels, pretrained=pretrained)#, pretrained=pretrained)
            if not pretrained:
                # Lógica para baixar os pesos e carregar no modelo.
                pretrained = "checkpoint/segformer.b1.512x512.ade.160k.pth"
                      
                pass
            #model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            #model.classifier[4] = nn.Conv2d(512, out_classes, kernel_size=1)
            return model
            
        return locals()

#print(architectures()["segformer"](3, 20, True))
