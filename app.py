import os
import io
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Lambda, Dense, Flatten
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, load_model
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing import image

np.random.seed()


# Load your trained model
vgg = VGG16(input_shape=[224,224] + [3], weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)
prediction = Dense(2, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)

model.load_weights('training_1/cp.ckpt')
# List all subdirectories (bird name folders) in the testing directory
bird_name_folders = ["Normal", 'Pneumonia']

# Streamlit app
st.title("Pneumonia Detecter")

# Upload a new image
uploaded_image = st.file_uploader("Upload an Xray", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    image_bytes = uploaded_image.getvalue()
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_data = preprocess_input(img_array)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction[0])

    # Get the predicted class name directly from the folder name
    predicted_class_name = bird_name_folders[predicted_class_index]

    # Display the prediction
    st.title(f"Diagnosis: {predicted_class_name}")