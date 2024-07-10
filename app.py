import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Load the Keras model
model = tf.keras.models.load_model('Model.h5')

# Streamlit app

def main():
    st.sidebar.subheader('Navigation')
    nav = st.sidebar.radio(' ', ['Home', 'About'])
    if nav == 'Home':
        st.header('Try out our AI Skin Disease Detector')
        st.write("Upload an image and check the skin disease you are sufferng.")

        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

            image = Image.open(uploaded_image)
            image_array = np.array(image)
            processed_image = preprocess_image(image_array)
            pred=np.argmax(model.predict(processed_image))

            label = {
                0:'Actinic keratoses',1:'Basal cell carcinoma',2:'Seborrhoeic Keratosis',
                3:'Dermatofibroma',4:'Melanocytic nevi',6:'Melanoma',5:'Vascular lesions'
             }
            disease=label[pred]
            params={'disease':disease,'val':pred}
            st.write("Prediction:")
            st.write(params)
    if nav== 'About' :
        st.write("THis project is done by Sriram")   

def preprocess_image(image_array):
    image_array=cv2.resize(image_array,(28,28))/255.0
    image=np.expand_dims(image_array,axis=0)
    return image

if __name__ == "__main__":
    main()
