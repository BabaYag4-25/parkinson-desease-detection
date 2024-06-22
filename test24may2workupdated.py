import numpy as np
import os
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import joblib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

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

# Fungsi untuk augmentasi data
def augment_data(audio_data, sampling_rate):
    augmented_data = []
    # Original data
    augmented_data.append(audio_data)
    # Add noise
    noise = np.random.normal(0, 0.005, audio_data.shape)
    augmented_data.append(audio_data + noise)
    # Pitch shifting
    pitch_shifted_up = librosa.effects.pitch_shift(audio_data, sr=sampling_rate, n_steps=2)
    augmented_data.append(pitch_shifted_up)
    pitch_shifted_down = librosa.effects.pitch_shift(audio_data, sr=sampling_rate, n_steps=-2)
    augmented_data.append(pitch_shifted_down)
    # Time stretching
    stretched_slow = librosa.effects.time_stretch(audio_data, rate=0.8)
    augmented_data.append(stretched_slow)
    stretched_fast = librosa.effects.time_stretch(audio_data, rate=1.2)
    augmented_data.append(stretched_fast)
    
    return augmented_data

# Fungsi untuk memuat data dan mengekstraksi fitur dengan augmentasi
def load_data_and_extract_features_with_augmentation(data_dir):
    features = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(data_dir, filename)
            audio_data, sampling_rate = librosa.load(filepath, sr=None)
            augmented_datas = augment_data(audio_data, sampling_rate)
            for augmented_data in augmented_datas:
                mfcc_features = extract_mfcc_features(augmented_data, sampling_rate)
                jitter = extract_jitter_features(augmented_data, sampling_rate)
                shimmer = extract_shimmer_features(augmented_data)
                combined_features = np.hstack((mfcc_features, jitter, shimmer))
                features.append(combined_features)
                if "parkinson" in filename:
                    labels.append(1)
                else:
                    labels.append(0)
    return np.array(features), np.array(labels)

# Fungsi untuk membangun model CNN
def build_enhanced_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, padding='same', kernel_size=3, input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.9), #SpatialDropout1D(0.6),
        tf.keras.layers.Conv1D(128, padding='same', kernel_size=3, kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.8), #SpatialDropout1D(0.6),
        tf.keras.layers.Conv1D(256, padding='same', kernel_size=3, kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.8), #SpatialDropout1D(0.6),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.9),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Memuat data latih dengan augmentasi
data_dir_parkinson_train = "E:/Universitas Brawijaya/SEMESTER 8/Tes Alat/Add Ons/Testing/Dataset Suara/parkinson fr"
data_dir_healthy_train = "E:/Universitas Brawijaya/SEMESTER 8/Tes Alat/Add Ons/Testing/Dataset Suara/healthy fr"
features_parkinson_train, labels_parkinson_train = load_data_and_extract_features_with_augmentation(data_dir_parkinson_train)
features_healthy_train, labels_healthy_train = load_data_and_extract_features_with_augmentation(data_dir_healthy_train)

# Menggabungkan data latih
features_train = np.concatenate((features_parkinson_train, features_healthy_train), axis=0)
labels_train = np.concatenate((labels_parkinson_train, labels_healthy_train), axis=0)

# Normalisasi data
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_train = features_train.reshape(features_train.shape[0], features_train.shape[1], 1)

# Konversi label menjadi one-hot encoding
labels_train = to_categorical(labels_train, num_classes=2)

# Bagi data menjadi data latih dan data uji
X_train, X_val, y_train, y_val = train_test_split(features_train, labels_train, test_size=0.2, random_state=42, stratify=labels_train)

# Membangun model
input_shape = (X_train.shape[1], 1)
model = build_enhanced_model(input_shape)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model_parkinson_augmenteddd2.h5", monitor='val_accuracy', save_best_only=True)

# Latih model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=250, batch_size=64, callbacks=[early_stopping, reduce_lr, model_checkpoint])

# Evaluasi model pada data uji
model = tf.keras.models.load_model("best_model_parkinson_augmenteddd2.h5")
evaluation = model.evaluate(X_val, y_val)
print(f"Validation Loss: {evaluation[0]}")
print(f"Validation Accuracy: {evaluation[1]}")

# Simpan scaler
joblib.dump(scaler, 'scaler_parkinson_combined_augmenteddd2.pkl')

# Plot Confusion Matrix
predictions = model.predict(X_val)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_val, axis=1)
cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Non-Parkinson', 'Parkinson'], yticklabels=['Non-Parkinson', 'Parkinson'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification Report
print(classification_report(true_labels, predicted_labels, target_names=['Non-Parkinson', 'Parkinson']))

# F1-Score
f1 = f1_score(true_labels, predicted_labels, average='weighted')
print(f"F1-Score: {f1}")
