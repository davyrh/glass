from keras.models import load_model
import keras
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import tensorflow as tf
st.title("Identification the Condition of Glasses of the House")
st.header("Architecture Image Detection")
st.text("Upload an image of any piece in  Glasses' house for Identification ")
from img_classification import teachable_machine_classification            
uploaded_file = st.file_uploader("Choose an image of glasses  in the house ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write(" Identifying the Glass ...")
    label = teachable_machine_classification(image, 'keras_model_glass.h5')
    if label == 0:
        st.write(" The glass is good ")
                          
    else:
        st.write("the glass is bad")
        
           
