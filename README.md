# Parkinson's Disease Detection Using CNN and Voice Features

## Overview
This project aims to detect Parkinson's disease using a machine learning model based on Convolutional Neural Networks (CNN). The system extracts Jitter and Shimmer features from voice recordings to classify whether a person has Parkinson's disease or not.

## Features
- **Feature Extraction:** Uses MFCC, Jitter, and Shimmer features from voice recordings.
- **Data Augmentation:** Applies noise addition, pitch shifting, and time stretching to improve model performance.
- **Model Training:** Uses a CNN model with multiple convolutional layers and batch normalization.
- **Real-time Prediction:** Provides a GUI interface for users to record their voice and get instant predictions.

## Project Structure
- **`Parkinson_Classification_JitterShimmer.py`**: Main script for feature extraction, data augmentation, model training, and evaluation.
- **`Load_Model.py`**: Loads and tests the trained model on new audio samples.
- **`GUI_Baru.py`**: Provides a Tkinter-based GUI for real-time voice-based Parkinson's disease detection.
- **`best_model_parkinson_augmented.h5`**: Trained CNN model for classification.
- **`scaler_parkinson_combined_augmented.pkl`**: Scaler used to normalize input features before feeding them into the model.

## Installation
1. Clone this repository.
2. Install required dependencies:
   ```sh
   pip install numpy librosa tensorflow joblib sklearn seaborn matplotlib pyaudio tkinter
   ```
3. Run the GUI application:
   ```sh
   python GUI_Baru.py
   ```

## How It Works
1. The user records their voice using the GUI.
2. The system extracts MFCC, Jitter, and Shimmer features.
3. The features are normalized and fed into the trained CNN model.
4. The model predicts whether the user has Parkinson's or not and displays the confidence score.

## Results & Performance
- Achieved an accuracy of **89%** on the validation dataset.
- Uses **data augmentation** to improve generalization.
- Provides an easy-to-use interface for non-technical users.

## Author
Developed by [Your Name] as part of a research project on Parkinson's disease detection using voice analysis and deep learning.
