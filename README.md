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
