import streamlit as st
import tensorflow as tf
import numpy as np


def model_prediction(testImage):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(testImage, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

st.set_page_config(
        page_title="Plant Disease predictor"
)



st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select page", ["Home", "About", "Disease recognition"])

if app_mode == "Home":
    st.header("Plant disease recognition system")
    st.markdown(
        """
    # 🌿 Plant Disease Detection App
    
    Welcome to the **Plant Disease Detection System** powered by **Convolutional Neural Networks (CNNs)**.  
    This tool helps farmers, researchers, and plant enthusiasts quickly identify potential diseases from leaf images.  
    
    ---
    ### 🚀 How it works:
    1. **Upload** a clear image of a plant leaf.  
    2. The CNN model will **analyze** the image.  
    3. Get a **prediction** of the plant’s health along with the confidence score.  
    
    ---
    ### 🌱 Why use this app?
    - Early detection helps prevent crop loss.  
    - AI-driven predictions are **fast and accurate**.  
    - Easy-to-use interface, no technical background required.  
    
    ---
    👉 Use the sidebar to navigate through the app and try out the disease detection feature!
    """
    )

elif app_mode == "About":
    st.header("About")
    st.markdown(
        """
    # ​ About This App

    This application is built using a rich and diverse dataset—the **New Plant Diseases Dataset**, sourced from [Kaggle by user vipoooool]. It features approximately **87,000** RGB images of both **healthy and diseased plant leaves**, encompassing **38 distinct classes**. :contentReference[oaicite:0]{index=0}

    The dataset provides a comprehensive foundation for training convolutional neural networks (CNNs) to accurately detect plant diseases from leaf imagery.

    ---
    ###  Dataset Highlights
    - **Size & Diversity**: ~87K images covering 38 classes, including a wide range of crops and disease types. :contentReference[oaicite:1]{index=1}
    - **Varied Conditions**: Includes images with different lighting, backgrounds, and leaf orientations—making the model more robust to real-world scenarios.

    ---
    ###  Why We Chose This Dataset
    - **Extensive Coverage**: Its breadth across species and disease types ensures the model learns nuanced visual characteristics.
    - **Proven Effectiveness**: Many implementations have used this data successfully in CNN-based plant disease detection tasks. :contentReference[oaicite:2]{index=2}
    - **Community Support**: It’s a well-known benchmark in the plant pathology and computer vision communities, used in tutorials, research, and competition settings.

    ---
    **Note:** While this dataset is excellent for training and benchmarking, users should still consult agricultural specialists for critical diagnostic decisions.

    Feel free to explore the model's performance, try uploading your own images for prediction, or check out our methodology for how the CNN processes your uploads!
    """
    )

elif app_mode == "Disease recognition":
    st.header("Disease recognition")
    test_image = st.file_uploader("Choose an image: ")
    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)
    if st.button("Make prediction"):
        with st.spinner("Making a prediction..."):
            result_index = model_prediction(test_image)
            class_names = [
                "Apple___Apple_scab",
                "Apple___Black_rot",
                "Apple___Cedar_apple_rust",
                "Apple___healthy",
                "Blueberry___healthy",
                "Cherry_(including_sour)___Powdery_mildew",
                "Cherry_(including_sour)___healthy",
                "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
                "Corn_(maize)___Common_rust_",
                "Corn_(maize)___Northern_Leaf_Blight",
                "Corn_(maize)___healthy",
                "Grape___Black_rot",
                "Grape___Esca_(Black_Measles)",
                "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                "Grape___healthy",
                "Orange___Haunglongbing_(Citrus_greening)",
                "Peach___Bacterial_spot",
                "Peach___healthy",
                "Pepper,_bell___Bacterial_spot",
                "Pepper,_bell___healthy",
                "Potato___Early_blight",
                "Potato___Late_blight",
                "Potato___healthy",
                "Raspberry___healthy",
                "Soybean___healthy",
                "Squash___Powdery_mildew",
                "Strawberry___Leaf_scorch",
                "Strawberry___healthy",
                "Tomato___Bacterial_spot",
                "Tomato___Early_blight",
                "Tomato___Late_blight",
                "Tomato___Leaf_Mold",
                "Tomato___Septoria_leaf_spot",
                "Tomato___Spider_mites_Two-spotted_spider_mite",
                "Tomato___Target_Spot",
                "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                "Tomato___Tomato_mosaic_virus",
                "Tomato___healthy",
            ]
        st.success("Model prediction: {}".format(class_names[result_index]))

