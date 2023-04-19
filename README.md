# Age Classification in Videos

This project aims to detect and classify faces in videos as either 'Adult' or 'Child' using deep learning techniques. The primary use case is to help tackle the issue of child pornography by identifying videos containing children.
Overview

The project consists of two main parts:

    Training a deep learning model for age classification based on images
    Analyzing videos in real-time and labeling faces as 'Adult' or 'Child' using the trained model

## Model Training

The model is based on the ResNet50 architecture, which has been pre-trained on the ImageNet dataset. A custom classification head is added to the base model to enable binary classification ('Adult' or 'Child'). The model is then trained on a large dataset of images containing both adult and child faces. Techniques such as data augmentation and early stopping have been employed to improve the model's performance and prevent overfitting.
Video Analysis

The video analysis part of the project uses OpenCV for face detection and the trained ResNet50 model for age classification. Detected faces in each video frame are passed through the predict_age_group function, which preprocesses the image, runs it through the model, and returns the age group prediction. Faces are then labeled as either 'Adult' or 'Child' and rectangles are drawn around them. The processed frames are saved into a new video file with the added annotations.
Dependencies

To run the project, the following dependencies are required:

    Python 3.6+
    TensorFlow 2.0+
    Keras
    OpenCV
    Numpy
    scikit-learn
