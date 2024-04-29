import streamlit as st
import subprocess

st.title('Right and Left Hand Detection')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Save the uploaded image temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Call the TensorFlow script using subprocess
    prediction = subprocess.check_output(["python", "tensorflow_script.py", "temp_image.jpg"])
    prediction = prediction.decode("utf-8").strip()

    # Display the uploaded image
    st.image("temp_image.jpg", caption='Uploaded Image', use_column_width=True)

    # Display the prediction
    st.write('Prediction:', prediction)
