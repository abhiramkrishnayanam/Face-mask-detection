# Face-mask-detection
Using CNN and Mediapipe
🧠 Face Mask Detection using CNN and MediaPipe
This project is a real-time face mask detection system that uses Convolutional Neural Networks (CNN) for classification and MediaPipe for facial detection. It helps identify whether a person is wearing a face mask or not using webcam input.

🔧 Technologies Used
Python

TensorFlow / Keras

MediaPipe

OpenCV

NumPy

Matplotlib (for visualization)

📁 Project Overview
Face Detection: MediaPipe detects faces in real time.

Preprocessing: Each detected face is resized and normalized.

Prediction: A trained CNN model determines if the face has a mask.

Output: Labels like "With Mask" or "Without Mask" are shown on the video feed with green/red bounding boxes.

🚀 How to Use
Install dependencies listed in the requirements.txt.

Ensure your webcam is connected.

Run the main Python script to start detection.

✅ Features
Real-time face mask detection

High accuracy CNN-based classifier

Visual feedback with bounding boxes and labels

User-friendly and customizable

🏗️ Model Training (Optional)
You can retrain the CNN model using your own dataset organized into two folders: with_mask and without_mask.

📌 Notes
Make sure the model.h5 file is present in the working directory.

If predictions are inaccurate, you may adjust the classification threshold or retrain the model with more data.

🙋‍♂️ Author
Abhiram
Machine Learning & Data Science Enthusiast
