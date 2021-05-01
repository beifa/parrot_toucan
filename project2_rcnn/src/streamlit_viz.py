import streamlit as st
import PIL.Image as Image
import evaluate
import cv2


st.title('Hello')

img = st.sidebar.selectbox(
    'Select image',
    ('232868961_44afff1a84_c.jpg', )

)

input_image = '../test_data/' + img
st.write('## Image: ')
image = Image.open(input_image)
st.image(image, width = 512)

click = st.button('Predict')
if click:
    image = cv2.cvtColor( cv2.imread(str(input_image)), cv2.COLOR_BGR2RGB)
    pred_img = evaluate.evl_streamlit(image)

    st.write('## Find parrot: ')
    st.image(pred_img, width = 512)