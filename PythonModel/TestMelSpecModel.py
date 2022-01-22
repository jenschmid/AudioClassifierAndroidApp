"""
This file contains code for testing the mel spectrogram model with a single audio from a file in wav format.
"""
import tensorflow as tf
import librosa

from Constants import cut_audio_pad_last, SAMPLE_RATE, N_MELS, IS_MONO

audio_path = "data/TestTracks/Speech.wav"  # The path to the audio that is loaded and used for testing
model_path = "model/melspectrogram/keras/"  # The path to the model that should be used

new_model = tf.keras.models.load_model(model_path)
audio, sample_rate_audio = librosa.load(audio_path, sr=SAMPLE_RATE, mono=IS_MONO)

audio_pieces = cut_audio_pad_last(audio)  # Cut the audio into pieces since the model can only handle fixed size pieces

for audio_piece in audio_pieces:
    melspect = librosa.feature.melspectrogram(y=audio_piece, sr=SAMPLE_RATE, n_mels=N_MELS)  # Compute a mel spectrogram of the audio piece
    melspect = melspect.reshape((1, N_MELS, 63))  # Reshape the array such that the model can use it

    prediction = new_model.predict(melspect)  # Predict the type of the audio and get scores for all categories
    print(prediction)  # TODO make better print statement that explains categories
