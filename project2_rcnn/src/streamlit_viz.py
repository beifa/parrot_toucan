import streamlit as st
import PIL.Image as Image
import evaluate
import cv2
import numpy as np
from pathlib import Path

from model import PT_RRCNN
from dataset import PT_test
from utils import collate_fn
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from evaluate import evl_streamlit, evl_streamlit_grid

PATH_TEST_IMG = '../project2_rcnn/input/test_data/'
PATH_MODEL = 'model_rcnn/fasterrcnn_resnet50_fpn.pth'

def make_model(data):

    model = PT_RRCNN(test=True)
    model.load_state_dict(torch.load(PATH_MODEL))

    data_img = PT_test(data)
    test_loader =  DataLoader(data_img,
                        batch_size = 1,
                        num_workers = 2,     
                        collate_fn = collate_fn
                        )
    return model, test_loader

image_logo = Image.open('../project2_rcnn/input/logo.jpg')
st.image(image_logo, use_column_width = True)


st.title('NN detection parrot on photos')

st.write(
    """
    ## Model (Faster_RCNN, place for seond model):

    ### Faster_RCNN:
        - Trained on 40 epoch 5 fold result ~ .80 IOU
          not used aurgumentation and lr_scheduler

    ### seond model:
        - blablaa
    
    ***
    """
)

img = st.sidebar.selectbox(
    'Select image',
    tuple(f.name for f in Path(PATH_TEST_IMG).glob('*'))
)

detection_threshold = 0.5
# bar terchhold
# score and neme for one image

input_image = '../project2_rcnn/input/test_data/' + img
st.header('Image: ')
image = Image.open(input_image)
st.image(image, width = 512)

checkbox = st.checkbox("Show All")
click = st.button('Predict')

if checkbox:    
    if click:
        test_img = []
        for img in Path(PATH_TEST_IMG).glob('*'):
            test_img.append(cv2.cvtColor( cv2.imread(str(img)), cv2.COLOR_BGR2RGB))

        rrcnn_model, d_loader = make_model(test_img) 
        out = evl_streamlit_grid(rrcnn_model, d_loader)
        tmp = []        
        for i in range(len(test_img)):
            im = test_img[i]
            # print(out[i])
            # print(out[i]['boxes'][0])
            # print(out[i]['boxes'])
            b = out[i][0]['boxes'].data.cpu().numpy()
            if len(b) > 0:
                s = out[i][0]['scores'].data.cpu().numpy()
                bx = b[s>=detection_threshold]
                if len(bx) > 0:
                    # for b in bx:
                    b = bx[0]
                    cv2.rectangle(im,
                                (b[0], b[1]),
                                (b[2], b[3]),
                                (0,0,255), 2)
            im = cv2.resize(im, (799,533))
            tmp.append(im)
        st.image(
            np.concatenate(np.array(tmp[:18]).reshape(3, 533*6,799,3), axis= 1)
        )       

else:
    if click:
        image = cv2.cvtColor( cv2.imread(str(input_image)), cv2.COLOR_BGR2RGB)
        rrcnn_model, d_loader = make_model([image]) 
        pred_img = evl_streamlit(rrcnn_model, d_loader, image, detection_threshold)
        st.write('## Find parrot: ')
        st.image(pred_img, width = 512)
   


