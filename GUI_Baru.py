import tkinter as tk
import pyaudio
import wave
import time
import numpy as np
import joblib

from keras.models import load_model
from tkinter import messagebox
from threading import Thread
from sklearn.preprocessing import StandardScaler
import librosa
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def extract_mfcc_features(audio_data, sampling_rate):
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=13)
    return mfcc_features.mean(axis=1)

def extract_jitter_features(audio_data, sampling_rate):
    pitches, _ = librosa.core.piptrack(y=audio_data, sr=sampling_rate)
    pitch_mean = np.mean(pitches, axis=0)
    jitter = np.var(np.diff(pitch_mean))
    return jitter

def extract_shimmer_features(audio_data):
    rms = librosa.feature.rms(y=audio_data)
    shimmer = np.var(rms)
    return shimmer

def resample_audio(audio_data, original_sr, target_sr=16000):
    if original_sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
    return audio_data

def extract_features(audio_file, target_sr=16000):
    audio_data, original_sr = librosa.load(audio_file, sr=None)
    audio_data = resample_audio(audio_data, original_sr, target_sr)
    mfcc_features = extract_mfcc_features(audio_data, target_sr)
    jitter = extract_jitter_features(audio_data, target_sr)
    shimmer = extract_shimmer_features(audio_data)
    combined_features = np.hstack((mfcc_features, jitter, shimmer))
    return combined_features

def normalize_data(X, scaler=None):
    if scaler is None:
        scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled.reshape(1, -1, 1), scaler

class ParkinsonDetectionApp:
    
    def __init__(self, master):
        self.master = master
        master.title("Sistem Deteksi Parkinson")
        master.geometry("650x420")
        master.configure(bg="#F1F3F4") 

        self.page1 = Page1(master, self)
        self.page2 = Page2(master, self)

        self.show_page1()

    def show_page1(self):
        self.page2.hide()
        self.page1.show()

    def show_page2(self, prediction_result, confidence):
        self.page1.hide()
        self.page2.show(prediction_result, confidence)

class Page1:
    def __init__(self, master, app):
        self.master = master
        self.app = app

        self.frame = tk.Frame(master, bg="#F1F3F4")
        self.frame.pack(expand=True)

        self.label = tk.Label(self.frame, text="Rekam suara anda", font=("Helvetica", 28, "bold"), bg="#F1F3F4", fg="#5F6368")
        self.record_button = tk.Button(self.frame, text="Mulai Rekam", font=("Helvetica", 20, "bold"), bg="#1A73E8", fg="#FFFFFF", command=self.record_audio, borderwidth=0, relief="flat")

        self.label.pack(pady=20)
        self.record_button.pack(pady=50)

    def show(self):
        self.frame.pack(expand=True)

    def hide(self):
        self.frame.pack_forget()

    def record_audio(self):
        for i in range(3, 0, -1):
            self.label.config(text=f"Perekaman dimulai dalam {i}")
            self.master.update()
            time.sleep(1)

        self.label.config(text="Perekaman suara...")
        self.master.update()

        recording_thread = Thread(target=self.simulate_recording)
        recording_thread.start()

    def simulate_recording(self):
        try:
            duration = 4
            file_name = "recorded_audio.wav"
            sample_rate = 16000  

            p = pyaudio.PyAudio()

            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=sample_rate,  
                            input=True,
                            frames_per_buffer=1024)

            frames = []

            for i in range(0, int(sample_rate / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            with wave.open(file_name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                wf.setframerate(sample_rate)  # Ubah laju sampel di sini
                wf.writeframes(b''.join(frames))

            self.label.config(text="Rekaman suara anda berhasil")
            self.master.update()

            model_path = "best_model_parkinson_augmented.h5"
            scaler_path = 'scaler_parkinson_combined_augmented.pkl'

            predicted_label, prediction = predict_new_audio(file_name, model_path, scaler_path)
            label_map = {0: "Non-Parkinson", 1: "Parkinson"}
            result = label_map[predicted_label[0]]
            confidence = prediction[0][predicted_label[0]]

            self.app.show_page2(result, confidence)

        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat merekam audio: {str(e)}")

class Page2:
    def __init__(self, master, app):
        self.master = master
        self.app = app

        self.frame = tk.Frame(master, bg="#F1F3F4")
        self.frame.pack(expand=True)

        self.label = tk.Label(self.frame, text="Hasil Prediksi:", font=("Helvetica", 28, "bold"), bg="#F1F3F4", fg="#5F6368")
        self.prediction_label = tk.Label(self.frame, text="", font=("Helvetica", 24), bg="#F1F3F4", fg="#5F6368")
        self.confidence_label = tk.Label(self.frame, text="", font=("Helvetica", 20), bg="#F1F3F4", fg="#5F6368")

        self.back_button = tk.Button(self.frame, text="Back", font=("Helvetica", 20, "bold"), bg="#1A73E8", fg="#FFFFFF", command=app.show_page1, borderwidth=0, relief="flat")

        self.label.pack(pady=20)
        self.prediction_label.pack(pady=10)
        self.confidence_label.pack(pady=10)
        self.back_button.pack(pady=30)

    def show(self, prediction_result, confidence):
        self.prediction_label.config(text=f"{prediction_result}")
        self.confidence_label.config(text=f"Confidence: {confidence * 100:.2f}%")
        self.frame.pack(expand=True)

    def hide(self):
        self.frame.pack_forget()

def predict_new_audio(audio_path, model_path, scaler_path):
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Ekstraksi fitur dari file audio baru
    features = extract_features(audio_path)
    
    # Normalisasi data
    features = features.reshape(1, -1)
    features, _ = normalize_data(features, scaler)
    
    # Prediksi
    prediction = model.predict(features)
    
    # Konversi prediksi ke label
    predicted_label = np.argmax(prediction, axis=1)
    
    return predicted_label, prediction

if __name__ == "__main__":
    root = tk.Tk()
    app = ParkinsonDetectionApp(root)
    root.mainloop()
