import cv2
import torch
import numpy as np
from pathlib import Path
import streamlit as st
import PIL.Image as Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import PT_RRCNN
from dataset import PT_test
from utils import collate_fn, plot_rectangle
from evaluate import evl_streamlit, evl_streamlit_grid, evl_streamlit_folds

PATH_TEST_IMG = '../project2_rcnn/input/test_data/'
PATH_MODEL = 'model_rcnn/fasterrcnn_resnet50_fpn.pth'

# @st.cache(allow_output_mutation=True)
def make_model(data: list):

    model = PT_RRCNN(test=True)
    model.load_state_dict(torch.load(PATH_MODEL))

    data_img = PT_test(data)
    test_loader =  DataLoader(data_img,
                        batch_size = 1,
                        num_workers = 2,     
                        collate_fn = collate_fn
                        )
    return model, test_loader


# @st.cache
def load_test_data()->list:
    # loads test images
    test_img = []
    for img in Path(PATH_TEST_IMG).glob('*'):
        test_img.append(cv2.cvtColor( cv2.imread(str(img)), cv2.COLOR_BGR2RGB))
    return test_img

# make logo
image_logo = Image.open('../project2_rcnn/input/logo.jpg')
st.image(image_logo, use_column_width = True)
st.title('Hey, we try find parrot on photos.')

# add description
bar_expander =st.beta_expander('Description:')
bar_expander.markdown("""
This is a simple model for finding parrots in a photo.
The model was trained on 94 photographs of parrots in the wild. 
For detect used(Faster_RCNN). 
***
* **Faster_RCNN**:
    Trained on 40 epoch one fold, result ~ .80 IOU not used aurgumentation and lr_scheduler
* **Faster_RCNN & 5_folds:**
    You see predict all folds

PS:
   Image 14 where fly parrot not detected, but we try select Faster_RCNN & 5_folds add.....
""")

st.sidebar.header('Settings:')


# to fix

# error: OpenCV(4.5.1) /tmp/pip-req-build-ddpkm6fn/opencv/modules/imgproc/src/color.cpp:182:
# error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'



# upload_img = st.sidebar.file_uploader('Upload your image JPE file', type = ['jpg'])

img = st.sidebar.selectbox(
    'Select image',
    tuple(f.name for f in Path(PATH_TEST_IMG).glob('*'))
    )

nn = st.sidebar.selectbox(
    'Model',
    ('Faster_RCNN', 'Faster_RCNN & 5_folds')
)

# threshold probability model
detection_threshold = st.sidebar.slider('Detection threshold', 0.1, 1.0, 0.5)

# error load image
# if upload_img is not None:
#     input_image = upload_img
# else:

input_image = '../project2_rcnn/input/test_data/' + img
st.header('Image: ')
image = Image.open(input_image)
st.image(image, width = 512)

click = st.button('Predict')
if nn == 'Faster_RCNN':
    checkbox = st.checkbox("Show All")
    if checkbox:    
        if click:
            # make Grid
            test_img = load_test_data()
            rrcnn_model, d_loader = make_model(test_img) 
            out = evl_streamlit_grid(rrcnn_model, d_loader)
            tmp = []        
            for i in range(len(test_img)):
                ori_image = test_img[i]
                im = plot_rectangle(out[i], ori_image, detection_threshold, (0,0,255), text=False)
                # resize for eq image       
                im = cv2.resize(im, (799,533))
                tmp.append(im)
            st.image(
                np.concatenate(np.array(tmp[:18]).reshape(3, 533*6,799,3), axis= 1)
            )       
    else:
        if click:
            # Single image
            image = cv2.cvtColor( cv2.imread(str(input_image)), cv2.COLOR_BGR2RGB)
            rrcnn_model, d_loader = make_model([image]) 
            pred_img = evl_streamlit(rrcnn_model, d_loader, image, detection_threshold)
            st.write('## Find parrot: ')
            st.image(pred_img, width = 512)
else:
    # Faster_RCNN & 5_folds
    if click:
        image = cv2.cvtColor( cv2.imread(str(input_image)), cv2.COLOR_BGR2RGB)
        rrcnn_model, d_loader = make_model([image]) 
        out = evl_streamlit_folds(rrcnn_model, d_loader)       
        color = [
                 (0,0,255),
                 (0,255,255),
                 (255,0,255),
                 (10,0,255),
                 (255,255,0)
                 ]
        outline_thickness = 1
        for i, d in enumerate(np.asarray(out)[:, 0]):
            im = plot_rectangle(d, image, detection_threshold, color[i], outline_thickness, text=False) 
        st.write('## Find parrot: ')
        st.image(im, width = 512)