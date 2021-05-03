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
from utils import collate_fn, plot_rectangle,set_seed

set_seed(13)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

PATH_TEST_IMG = '../project2_rcnn/input/test_data/'
PATH_MODEL = 'model_rcnn/fasterrcnn_resnet50_fpn.pth'
PATH_MODE_FOLDS = 'model_rcnn/folds'

def evaluate(model, loader)->list:
    """
    model: rrcnn_resnet50
    loader : dataloader
    make predict
    return array    
    """   

    model.load_state_dict(torch.load(PATH_MODEL))
    model.eval()
    model.to(device)
    with torch.no_grad():
        out = []
        for images,i in loader:  
            image = list(image.to(device) for image in images)
            outputs = model(image)
            out.append(outputs)
    return out 

@st.cache
def evl_streamlit(model, loader, ori_image : list, threshold : int)->list:  
    """
    model: rrcnn_resnet50
    loader : dataloader
    ori_image : list,  origin image not changed
    return image after plot prediction box
    """
    model.eval()
    model.to(device)
    with torch.no_grad(): 
      for images, i in loader:
        image = list(image.to(device) for image in images) 
        out = model(image)
        im = plot_rectangle(out, ori_image, threshold, (0,0,255), text=True)             
        return im

def evl_streamlit_grid(model, loader)->list:
    """
    model: rrcnn_resnet50
    loader : dataloader
    return list for grid
    """
    model.eval()
    model.to(device)
    with torch.no_grad(): 
        out = []
        for images, i in loader:
            image = list(image.to(device) for image in images) 
            outputs = model(image)
            out.append(outputs)
    return out

# @st.cache(hash_funcs={torch.Tensor: evl_streamlit_folds})
def evl_streamlit_folds(model, loader)->list:
    """
    model: rrcnn_resnet50
    loader : dataloader
    return: list, predict for single image but for each fold(5)
    """    
    model.eval()
    model.to(device)
    tmp_out = []
    for f in range(5):
        model.load_state_dict(torch.load(Path(PATH_MODE_FOLDS) / f'fasterrcnn_resnet50_fpn_{f}.pth'))
        with torch.no_grad():
            out = []
            for images,i in loader:  
                images = list(image.to(device) for image in images)
                outputs = model(images)
                out.append(outputs)
        tmp_out.append(out)                    
    return tmp_out 