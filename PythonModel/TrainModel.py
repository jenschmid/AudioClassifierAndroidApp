import librosa
import os
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import RMSprop

from Constants import SAMPLE_RATE, AUDIO_PIECE_LENGTH, IS_MONO, N_MELS, cut_audio

num_epochs = 50
batch_size = 70
path = "data/data2"

all_tracks = []
genre = []
list_matrices = []

for cls in os.listdir(path):
    for sound in tqdm(os.listdir(os.path.join(path, cls))):


        y, sr = librosa.load(os.path.join(os.path.join(path, cls, sound)), sr=SAMPLE_RATE, mono=IS_MONO)
        song_pieces = cut_audio(y)
        for song_piece in song_pieces:
            melspect = librosa.feature.melspectrogram(y=song_piece, sr=SAMPLE_RATE, n_mels=N_MELS)
            list_matrices.append(song_piece.reshape((AUDIO_PIECE_LENGTH, 1)))
            all_tracks.append(melspect)
            genre.append(cls)

encoder = LabelBinarizer()
labels = encoder.fit_transform(np.array(genre))


#Train spectrogram classifier
X_train, X_test, y_train, y_test = train_test_split(list_matrices, labels, test_size=0.33, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)
X_val = np.array(X_val)

model = models.Sequential()
model.add(layers.Conv1D(32, 4, activation='relu', input_shape=(AUDIO_PIECE_LENGTH, 1)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 3, activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation="softmax"))

model.summary()


model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=RMSprop(learning_rate=0.001),  metrics='accuracy')

history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val))

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

tf.saved_model.save(model, "model/spectrogram")
tf.keras.models.save_model(model, "model/spectrogram/keras")

print("Test Loss: ")
print(test_loss)
print("Test Accuracy: ")
print(test_acc)


#Train melspectrogram classifier
X_train2, X_test2, y_train2, y_test2 = train_test_split(all_tracks, labels, test_size=0.33, random_state=42)
X_val2, X_test2, y_val2, y_test2 = train_test_split(X_test2, y_test2, test_size=0.5, random_state=42)



X_train2 = np.array(X_train2)
X_test2 = np.array(X_test2)
X_val2 = np.array(X_val2)

model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 63, 1)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))

model2.add(layers.Flatten())
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(3, activation="softmax"))

model2.summary()

model2.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=RMSprop(learning_rate=0.001),  metrics='accuracy')


model2.summary()
#X_train = np.array(X_train).reshape(1,-1)
#y_train = y_train.reshape(1,-1)

history2 = model2.fit(X_train2, y_train2, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val2, y_val2))

test_loss2, test_acc2 = model2.evaluate(X_test2,  y_test2, verbose=2)

tf.saved_model.save(model2, "model/melspectrogram")
tf.keras.models.save_model(model2, "model/melspectrogram/keras")

print("Test Loss: ")
print(test_loss2)
print("Test Accuracy: ")
print(test_acc2)