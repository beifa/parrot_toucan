import cv2
import torch
import torchvision
import torch.nn as nn
import numpy as np
from pathlib import Path
import PIL.Image as Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

from model import PT_RRCNN
from dataset import PT_test
from utils import collate_fn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# import pytorch_lightning as pl
# pl.seed_everything(13)


PATH_TEST_IMG = '../project2_rcnn/input/test_data/'
PATH_MODEL = 'model_rcnn/fasterrcnn_resnet50_fpn.pth'

def evaluate(model, loader)->list:   

    model.load_state_dict(torch.load(PATH_MODEL))
    model.eval()
    model.to(device)
    with torch.no_grad():
        out = []
        for images,i in loader:  
            image = list(image.float().to(device) for image in images)
            outputs = model(image)
            out.append(outputs)
    return out 

def evl_streamlit(model, loader, ori_image, threshold)->list:  
    model.eval()
    model.to(device)
    with torch.no_grad(): 
      for images, i in loader:
        image = list(image.float().to(device) for image in images) 
        out = model(image)    
        
        im = ori_image
        b = out[0]['boxes'].data.cpu().numpy()
        if len(b) > 0:
            s = out[0]['scores'].data.cpu().numpy()        
            bx = b[s>=threshold]
            if len(bx) > 0:
                b = bx[0]
                cv2.rectangle(im,
                        (b[0], b[1]),
                        (b[2], b[3]),
                        (0,0,255), 3) 
                cv2.putText(im,
                        str(round(s[0], 2)),
                        org=(int((b[0]+b[2])/2), 
                             int(b[1]- 40)
                             ),
                        fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale = 1.05,
                        color = (255, 255, 255),
                        thickness = 2
                        )                
                return im
            else: return img
        else: return img


def evl_streamlit_grid(model, loader)->list:
    model.eval()
    model.to(device)
    with torch.no_grad(): 
        out = []
        for images, i in loader:
            image = list(image.float().to(device) for image in images) 
            outputs = model(image)
            out.append(outputs)
    return out


if __name__ == "__main__":  
    test_img = []
    detection_threshold = 0.5

    model = PT_RRCNN(test=True)
    for img in Path(PATH_TEST_IMG).glob('*.*'):
        test_img.append(cv2.cvtColor( cv2.imread(str(img)), cv2.COLOR_BGR2RGB)) 

    print('Test data size: ', len(test_img))
    dataset_test = PT_test(test_img)
    test_loader =  DataLoader(dataset_test,
                               batch_size = 1,  
                               num_workers = 2, 
                               collate_fn = collate_fn
                            )

    out = evaluate(model, test_loader)
    plt.figure(figsize=(20,10)) 

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