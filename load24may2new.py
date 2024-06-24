import numpy as np
import os
import librosa
import joblib
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Fungsi untuk ekstraksi fitur MFCC
def extract_mfcc_features(audio_data, sampling_rate):
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=13)
    return mfcc_features.mean(axis=1)

# Fungsi untuk ekstraksi fitur Jitter
def extract_jitter_features(audio_data, sampling_rate):
    pitches, _ = librosa.core.piptrack(y=audio_data, sr=sampling_rate)
    pitch_mean = np.mean(pitches, axis=0)
    jitter = np.var(np.diff(pitch_mean))
    return jitter

# Fungsi untuk ekstraksi fitur Shimmer
def extract_shimmer_features(audio_data):
    rms = librosa.feature.rms(y=audio_data)
    shimmer = np.var(rms)
    return shimmer

# Fungsi untuk memuat data uji dan mengekstraksi fitur berdasarkan subfolder
def load_and_extract_test_features(data_dir_test):
    features_test = []
    labels_test = []
    filenames = []
    for root, dirs, files in os.walk(data_dir_test):
        for filename in files:
            if filename.endswith(".wav"):
                label = 1 if 'parkinson' in root.lower() else 0
                filepath = os.path.join(root, filename)
                audio_data, sampling_rate = librosa.load(filepath, sr=None)
                mfcc_features = extract_mfcc_features(audio_data, sampling_rate)
                jitter = extract_jitter_features(audio_data, sampling_rate)
                shimmer = extract_shimmer_features(audio_data)
                combined_features = np.hstack((mfcc_features, jitter, shimmer))
                features_test.append(combined_features)
                filenames.append(filename)
                labels_test.append(label)
    return np.array(features_test), filenames, np.array(labels_test)

# Fungsi untuk melakukan prediksi dan evaluasi
def classify_test_data(model_path, scaler_path, data_dir_test):
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    features_test, filenames, labels_test = load_and_extract_test_features(data_dir_test)
    features_test = scaler.transform(features_test)
    features_test = features_test.reshape(features_test.shape[0], features_test.shape[1], 1)
    predictions = model.predict(features_test)
    predicted_labels = np.argmax(predictions, axis=1)

    for i, prediction in enumerate(predictions):
        print("File:", filenames[i])
        if predicted_labels[i] == 1:
            print("Hasil Klasifikasi: Parkinson")
        else:
            print("Hasil Klasifikasi: Sehat")

    cm = confusion_matrix(labels_test, predicted_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Parkinson'], yticklabels=['Healthy', 'Parkinson'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print(classification_report(labels_test, predicted_labels, target_names=['Healthy', 'Parkinson']))
    print("F1-Score:", f1_score(labels_test, predicted_labels, average='weighted'))

# Path model dan scaler yang telah disimpan
model_path = "best_model_parkinson_augmenteddd2.h5"
scaler_path = "scaler_parkinson_combined_augmenteddd2.pkl"

# Path data uji
data_dir_test = "E:/Universitas Brawijaya/SEMESTER 8/Tes Alat/Add Ons/Testing/Uji/"

# Melakukan prediksi
classify_test_data(model_path, scaler_path, data_dir_test)
