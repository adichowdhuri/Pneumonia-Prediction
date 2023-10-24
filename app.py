import os
import io
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Lambda, Dense, Flatten
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, load_model
import tensorflow as tf
import streamlit as st
from PIL import Image as pil_image
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="dAIgnostix", page_icon="👨‍⚕️")


# Load your trained model
inception = InceptionV3(input_shape=[224,224] + [3], weights='imagenet', include_top=False)
for layer in inception.layers:
    layer.trainable = False
x = Flatten()(inception.output)
prediction = Dense(2, activation='softmax')(x)
model = Model(inputs=inception.input, outputs=prediction)

model.load_weights('training_1/cp.ckpt')
# List all subdirectories (bird name folders) in the testing directory
folders = ["Normal", 'Pneumonia']


import base64

def get_image_base64_str(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

icon_path = "Picture1.png"
image_base64_str = get_image_base64_str(icon_path)
st.markdown(
    f'<div style="text-align: center;"><img src="data:image/png;base64,{image_base64_str}" style="max-width: 100%;"></div>', 
    unsafe_allow_html=True
)

# CSS for green text
st.markdown("""
<style>
    .green-text {
        color: green;
    }
</style>
<div style="text-align: center; font-size: 48px; font-weight: bold;">d<span class="green-text">AI</span>gnostix</div>
<div style="text-align: center; font-size: 24px; margin-top: 5px;">Your precision diagnostic partner</div>
            
""", unsafe_allow_html=True)


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