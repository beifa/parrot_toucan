import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class PT_RRCNN(nn.Module):

    def __init__(self, num_classes: int = 2, test :bool = False):
        super().__init__()
        self.test = test
        if self.test:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                              pretrained_backbone=False)
        else:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    def forward(self, x, y = None):
        if self.test:      
            return self.model(x)
        else:
            assert y is not None, f'target error PT_RRCNN, y : {y}'
            return self.model(x, y)