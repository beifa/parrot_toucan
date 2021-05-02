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

import streamlit as st

from model import PT_RRCNN
from dataset import PT_test
from utils import collate_fn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# import pytorch_lightning as pl
# pl.seed_everything(13)


PATH_TEST_IMG = '../project2_rcnn/input/test_data/'
PATH_MODEL = 'model_rcnn/fasterrcnn_resnet50_fpn_not_find_0ne_001sgdnotshel.pth'#fasterrcnn_resnet50_fpn.pth'
PATH_MODE_FOLDS = 'model_rcnn/folds'

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

@st.cache
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
            else: return ori_image
        else: return ori_image


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

# @st.cache(hash_funcs={torch.Tensor: evl_streamlit_folds})
def evl_streamlit_folds(model, loder)->list:
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,  pretrained_backbone=False)
    model.eval()
    model.to(device)
    tmp_out = []
    for f in range(5):
        model.load_state_dict(torch.load(Path(PATH_MODE_FOLDS) / f'fasterrcnn_resnet50_fpn_{f}.pth'))
        with torch.no_grad():
            out = []
            for images,i in loder:  
                images = list(image.float().to(device) for image in images)
                outputs = model(images)
                out.append(outputs)
        tmp_out.append(out)                    
    return tmp_out 