"""
This file contains the model specification for two different tensorflow models.
The first model works with signal data (directly from wav files).
The second model works with mel spectrogram data (generated with the Librosa library).
The model definitions can be adapted if needed and the training parameter can be changed as well.

The data is taken from the /data folder.
Both models are trained at the same run and with the same data since data preparation takes a lot of time.

The two models are the stored in the /model folder and can be further processed by the other scripts.
"""
import librosa
import os
import numpy as np
from keras.layers import Dropout

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow.keras import layers, models

from Constants import SAMPLE_RATE, AUDIO_PIECE_LENGTH, IS_MONO, N_MELS, cut_audio

RANDOM_STATE = 42

# -------------------- Preparation of the data set --------------------
path_to_training_data = "data/data_long"  # Either the small or large data set can be used for the training

all_labels = []
all_audios_as_signal = []
all_audios_as_mel_spec = []

for label in os.listdir(path_to_training_data):
    # Loop through all three folders of training data
    for audio_track in tqdm(os.listdir(os.path.join(path_to_training_data, label))):
        # Loop through all audio files
        audio_signal, sample_rate_audio = librosa.load(os.path.join(os.path.join(path_to_training_data, label, audio_track)), sr=SAMPLE_RATE, mono=IS_MONO)
        audio_pieces = cut_audio(audio_signal)  # Cut all audios to the same length
        for audio_piece in audio_pieces:
            mel_spectrogram = librosa.feature.melspectrogram(y=audio_piece, sr=SAMPLE_RATE, n_mels=N_MELS) # Calculate the melspectrogram
            all_audios_as_mel_spec.append(mel_spectrogram)

            audio_piece = audio_piece.reshape((AUDIO_PIECE_LENGTH, 1)) # Reshape the array such that the model can use it
            all_audios_as_signal.append(audio_piece)

            all_labels.append(label)

encoder = LabelBinarizer()
labels = encoder.fit_transform(np.array(all_labels))  # One-hot encoding of all labels

# ----------------------------------------------------------------------
# ---------------------- MODEL WITH SIGNAL INPUT -----------------------
# ----------------------------------------------------------------------

num_epochs_signal = 100  # The number of epochs that is used for training (taken from the paper)
batch_size_signal = 100  # The batch size that is used for training (taken from the paper)

# Train test split of the data
x_train_signal, x_test_signal, y_train_signal, y_test_signal = train_test_split(all_audios_as_signal, labels, test_size=0.33, random_state=RANDOM_STATE)
x_val_signal, x_test_signal, y_val_signal, y_test_signal = train_test_split(x_test_signal, y_test_signal, test_size=0.5, random_state=RANDOM_STATE)

# Wrap the arrays, otherwise they cannot be used for training
x_train_signal = np.array(x_train_signal)
x_test_signal = np.array(x_test_signal)
x_val_signal = np.array(x_val_signal)

# The model definition
# Code adapted from https://github.com/Logan97117/environmental_sound_classification_1DCNN
model_signal = models.Sequential()
model_signal.add(layers.Conv1D(filters=16, kernel_size=64, strides=2, input_shape=(AUDIO_PIECE_LENGTH, 1)))
model_signal.add(layers.Activation(activation='relu'))
model_signal.add(layers.BatchNormalization())
model_signal.add(layers.MaxPooling1D(pool_size=8, strides=8))
model_signal.add(layers.Activation(activation='relu'))
model_signal.add(layers.Conv1D(filters=32, kernel_size=32, strides=2))
model_signal.add(layers.Activation(activation="relu"))
model_signal.add(layers.BatchNormalization())
model_signal.add(layers.MaxPooling1D(pool_size=8, strides=8))
model_signal.add(layers.Activation(activation="relu"))
model_signal.add(layers.Conv1D(filters=64, kernel_size=16, strides=2))
model_signal.add(layers.Activation(activation="relu"))
model_signal.add(layers.BatchNormalization())
model_signal.add(layers.Conv1D(filters=128, kernel_size=8, strides=2))
model_signal.add(layers.Activation(activation="relu"))
model_signal.add(layers.BatchNormalization())
model_signal.add(layers.Conv1D(filters=256, kernel_size=4, strides=2))
model_signal.add(layers.Activation(activation="relu"))
model_signal.add(layers.BatchNormalization())
model_signal.add(layers.MaxPooling1D(pool_size=4, strides=4))
model_signal.add(layers.Activation(activation="relu"))
model_signal.add(layers.Flatten())
model_signal.add(layers.Dense(64, activation='relu'))
model_signal.add(layers.Dense(3, activation="softmax"))

model_signal.summary()

# Additional parameters, also taken from the paper
model_signal.compile(loss=tf.keras.losses.MeanSquaredLogarithmicError(), optimizer=tf.keras.optimizers.Adadelta(), metrics='accuracy')

history_signal = model_signal.fit(x_train_signal, y_train_signal, epochs=num_epochs_signal, batch_size=batch_size_signal, validation_data=(x_val_signal, y_val_signal))

test_loss_signal, test_acc_signal = model_signal.evaluate(x_test_signal, y_test_signal, verbose=2)

print("Test Loss: ")
print(test_loss_signal)
print("Test Accuracy: ")
print(test_acc_signal)

# Saving two versions of the model
tf.saved_model.save(model_signal, "model/signal")
tf.keras.models.save_model(model_signal, "model/signal/keras")

# -------------------------------------------------------------------------
# -------------------- MODEL WITH MEL SPECTROGRAM INPUT -------------------
# -------------------------------------------------------------------------

num_epochs_melspec = 50  # The number of epochs that is used for training (taken from the paper)
batch_size_melspec = 64  # The batch size that is used for training (taken from the paper)

# Train test split of the data
x_train_melspec, x_test_melspec, y_train_melspec, y_test_melspec = train_test_split(all_audios_as_mel_spec, labels, test_size=0.33, random_state=RANDOM_STATE)
x_val_melspec, x_test_melspec, y_val_melspec, y_test_melspec = train_test_split(x_test_melspec, y_test_melspec, test_size=0.5, random_state=RANDOM_STATE)

# Wrap the arrays, otherwise they cannot be used for training
x_train_melspec = np.array(x_train_melspec)
x_test_melspec = np.array(x_test_melspec)
x_val_melspec = np.array(x_val_melspec)

# The model definition (taken from the paper)
model_melspec = models.Sequential()
model_melspec.add(layers.Conv2D(24, (6, 6), activation='relu', input_shape=(128, 63, 1)))
model_melspec.add(layers.MaxPooling2D((4, 2), strides=2))
model_melspec.add(layers.Conv2D(48, (5, 5), activation='relu'))
model_melspec.add(layers.MaxPooling2D((4, 2), strides=2))
model_melspec.add(layers.Conv2D(48, (5, 5), activation='relu'))
model_melspec.add(layers.Conv2D(60, (4, 4), activation='relu'))
model_melspec.add(layers.Conv2D(72, (4, 4), activation='relu'))
model_melspec.add(layers.Flatten())
model_melspec.add(layers.Dense(84, activation='relu'))
model_melspec.add(Dropout(0.5))
model_melspec.add(layers.Dense(3, activation="softmax"))

model_melspec.summary()

# Additional parameters, also taken from the paper
model_melspec.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics='accuracy')

history_melspec = model_melspec.fit(x_train_melspec, y_train_melspec, epochs=num_epochs_melspec, batch_size=batch_size_melspec, validation_data=(x_val_melspec, y_val_melspec))

test_loss_melspec, test_acc_melspec = model_melspec.evaluate(x_test_melspec, y_test_melspec, verbose=2)

print("Test Loss: ")
print(test_loss_melspec)
print("Test Accuracy: ")
print(test_acc_melspec)

# Saving two versions of the model
tf.saved_model.save(model_melspec, "model/melspectrogram")
tf.keras.models.save_model(model_melspec, "model/melspectrogram/keras")
