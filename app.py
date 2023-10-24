import os
import io
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Lambda, Dense, Flatten
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, load_model
import tensorflow as tf
import streamlit as st
from PIL import Image as pil_image
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="dAIgnostix", page_icon="👨‍⚕️")


# Load your trained model
vgg = VGG16(input_shape=[224,224] + [3], weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)
prediction = Dense(2, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)

model.load_weights('training_1/cp.ckpt')
# List all subdirectories (bird name folders) in the testing directory
folders = ["Normal", 'Pneumonia']

# Streamlit app
icon_path = "Picture1.png"
st.markdown(f'<div style="text-align: center;"><img src="{icon_path}" style="max-width: 100%;"></div>', unsafe_allow_html=True) 

# Upload a new image
uploaded_image = st.file_uploader("Upload an Xray", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image_stream = io.BytesIO(uploaded_image.getvalue())

    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make predictions on the uploaded image
    img = pil_image.open(image_stream).resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction[0])
    predicted_probability = prediction[0][predicted_class_index] * 100   # Convert to percentage


    # Get the predicted class name directly from the folder name
    predicted_class_name = folders[predicted_class_index]

    # Display the prediction
    st.title(f"Diagnosis: {predicted_class_name} (Probability: {predicted_probability}%)")