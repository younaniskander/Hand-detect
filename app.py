import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('cifar10_classification_model.h5')

model = load_model()

# Define class labels for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image).resize((32, 32))  # Resize image to match model input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return img_array

# Function to make predictions
def predict(image):
    img_array = preprocess_image(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    return class_names[predicted_class], confidence

# Main function
def main():
    st.title('CIFAR-10 Image Classifier')
    st.write('Upload an image of one of the following classes:')
    st.write(class_names)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make predictions
        class_name, confidence = predict(uploaded_file)
        st.write(f'Predicted Class: {class_name}')
        st.write(f'Confidence: {confidence:.2f}')

if __name__ == '__main__':
    main()
