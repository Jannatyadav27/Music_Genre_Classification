import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from matplotlib import pyplot as plt
from tensorflow.image import resize # type: ignore

# Function to load model
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("Trained_model.keras")
    return model

# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    # Chunk processing
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)

# Tensorflow Model Prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

# Sidebar
st.sidebar.markdown(
    """
    <style>
    /* Custom style to change the sidebar title color */
    .sidebar .sidebar-content .css-1d391kg {
        color: black !important;  /* Ensure the title is black */
    }
    </style>
    """, unsafe_allow_html=True)


st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project"])

# Main Page: Home
if app_mode == "Home":
    st.markdown(
    """
    <style>
    /* Global styles */
    .stApp {
        background-color: #2c3e50;  /* Dark Background */
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1 {
        color: black;
    }

    h2 {
        color: #ecf0f1;
    }

    /* Button styles */
    .stButton>button, .stFileUploader>div>button {
        background-color: #e74c3c;  /* Button background */
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover, .stFileUploader>div>button:hover {
        transform: scale(1.1);
        background-color: #c0392b;
    }

    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #34495e;  /* Sidebar Color */
        transition: all 0.3s ease;
    }

    .sidebar .sidebar-content:hover {
        background-color: #16a085;
    }

    /* 3D Animation for Sidebar */
    .sidebar .sidebar-content {
        transform: rotateY(0deg);
        transition: transform 0.5s;
    }
    .sidebar .sidebar-content:hover {
        transform: rotateY(10deg);
    }

    /* Navbar - Sticky with color transition */
    .navbar {
        position: sticky;
        top: 0;
        background-color: #2c3e50;
        color: white;
        padding: 10px;
        transition: background-color 0.3s ease-in-out;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .navbar:hover {
        background-color: #16a085;
    }

    /* Animation for title */
    @keyframes fadeIn {
        0% {
            opacity: 0;
            transform: translateY(-50px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    h1 {
        animation: fadeIn 1s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Music Genre Classification 🎶🎧")

    # Audio upload and prediction section
    st.subheader("Upload your music file to predict its genre:")

    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    
    if test_mp3 is not None:
        filepath = 'Test_Music/'+test_mp3.name
        
        # Play Button
        if st.button("Play Audio"):
            st.audio(test_mp3)
        
        # Predict Button
        if st.button("Predict"):
            with st.spinner("Please Wait.."):
                X_test = load_and_preprocess_data(filepath)
                result_index = model_prediction(X_test)
                st.balloons()
                label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
                st.markdown(f"**🎵 It's a [**{label[result_index]}**] music genre!**")

# About Page: Project Details
elif app_mode == "About Project":
    st.markdown("""
    <style>
    .stApp {
        background-color: #34495e;  /* Dark Background */
        color: #ffffff;
    }
    h1 {
        color: black;
    }

    h2 {
        color: #ecf0f1;
    }
    
    .stMarkdown {
        font-size: 16px;
        line-height: 1.8;
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("About the Project")

    st.markdown("""
    ### Music Genre Classification System

    This project uses deep learning algorithms to classify music genres based on audio files. The model is trained using a well-known dataset (GTZAN) consisting of 10 genres:
    - blues
    - classical
    - country
    - disco
    - hiphop
    - jazz
    - metal
    - pop
    - reggae
    - rock

    ### Project Overview:
    1. **Audio Preprocessing:** 
        The audio files are preprocessed by converting them into Mel spectrograms, which represent the audio in a visual format suitable for deep learning models.
    
    2. **Model:** 
        A Convolutional Neural Network (CNN) is used to classify the audio based on the Mel spectrogram images.

    3. **Dataset:** 
        The dataset contains 100 audio files per genre, each lasting 30 seconds. These are divided into smaller chunks for better model performance.

    4. **Implementation:** 
        The model is built and trained using TensorFlow/Keras, and it is used to classify a user's uploaded audio into one of the predefined genres.

    ### Key Features:
    - **Highly Accurate**: Based on state-of-the-art deep learning models.
    - **Fast Processing**: Quick audio classification within seconds.
    - **User-Friendly**: Easy-to-use interface for uploading and getting predictions.

    ### Dataset Details:
    - **GTZAN Dataset**: This dataset contains 10 different genres, with each genre having 100 tracks.
    - **Mel Spectrograms**: Audio features are extracted and transformed into Mel spectrograms for classification.

    ### Technologies Used:
    - Python
    - TensorFlow
    - Librosa
    - Streamlit
    - Keras

    This system can be used by music enthusiasts, researchers, or developers interested in audio classification and machine learning.
    """)
