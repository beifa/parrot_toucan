import cv2
import torch
import torchvision
import torch.nn as nn
import numpy as np
from pathlib import Path
import PIL.Image as Image
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# import pytorch_lightning as pl
# pl.seed_everything(13)

PATH_TEST_IMG = '../progect2_rcnn/test_data'
PATH_MODEL = '../progect2_rcnn/model_rcnn/fasterrcnn_resnet50_fpn.pth'

def collate_fn(batch):
    return tuple(zip(*batch))

class PT_test(Dataset):

    def __init__(self, data, transforms = None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
         
        img = self.data[idx]
        img = img / 255.0
        img = np.transpose(img, (2,0,1))

        if self.transforms is not None:
            img = self.transforms(img)        
        return torch.tensor(img, dtype = torch.float32) , idx

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

def evaluate()->list:
    
    model = PT_RRCNN(test=True)
    model.load_state_dict(torch.load(PATH_MODEL))
    model.eval()
    model.to(device)
    with torch.no_grad():
        out = []
        for images,i in test_loader:  
            images = list(image.float().to(device) for image in images)
            outputs = model(images)
            out.append(outputs)
    return out 


if __name__ == "__main__":  
    test_img = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for img in Path(PATH_TEST_IMG).glob('*.*'):
        test_img.append(cv2.cvtColor( cv2.imread(str(img)), cv2.COLOR_BGR2RGB)) 

    dataset_test = PT_test(test_img)
    test_loader =  DataLoader(dataset_test,
                            batch_size=1,    
                            num_workers=2,   
                            collate_fn=collate_fn
                            )

    out = evaluate()

    plt.figure(figsize=(20,10)) 

    detection_threshold = 0.5

    for i in range(1, len(test_img)):
        plt.subplot(5, 5, i) 
        im = test_img[i-1]
        b = out[i-1][0]['boxes'].data.cpu().numpy()
        if len(b) > 0:
            s = out[i-1][0]['scores'].data.cpu().numpy()        
            bx = b[s>=detection_threshold]
            if len(bx) > 0:
                bx = bx[0]
                cv2.rectangle(im,
                    (bx[0], bx[1]),
                    (bx[2], bx[3]),
                    (0,0,255), 3)
                plt.imshow(im)
                plt.axis('off')
            else:
                plt.imshow(im)
                plt.axis('off')

        else:
            plt.imshow(im)
            plt.axis('off')
    
    plt.axis('off')
    plt.show()

