# Emotion-Recognition
This project involves building an emotion recognition system using a Convolutional Neural Network (CNN) model trained on the https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset from Kaggle. The goal is to identify emotions such as angry, happy, sad, surprise, fear, and disgust from grayscale images of faces. Additionally, the system can suggest the emotion of a person in real-time using a webcam.

Project Structure

1.Data Preparation
Load and preprocess the dataset using ImageDataGenerator.
Split the dataset into training and validation sets.

2.Model Architecture
Construct a CNN model with multiple convolutional layers, batch normalization, activation functions, pooling layers, and dropout for regularization.
Use Adam optimizer for training.

3.Training
Compile the model with categorical cross-entropy loss and accuracy metrics.
Implement callbacks like ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau for efficient training.

4.Evaluation
Plot training and validation loss and accuracy to evaluate model performance.
Save the trained model for real-time emotion detection.

5.Real-Time Emotion Detection
Use OpenCV to capture video from the webcam.
Preprocess the frames and predict the emotion using the trained model.
