"""
This file contains code for testing the Spectrogram model with a single audio from a file in wav format.
"""
import tensorflow as tf
import librosa

from Constants import SAMPLE_RATE, cut_audio_pad_last, AUDIO_PIECE_LENGTH

audio_path = "data/song/en001a.wav"  # The path to the audio that is loaded and used for testing

new_model = tf.keras.models.load_model('model/spectrogram/keras/')  # The path to the model that should be used

audio, sample_rate_audio = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)

audio_pieces = cut_audio_pad_last(audio)  # Cut the audio into pieces since the model can only handle fixed size pieces

for audio_piece in audio_pieces:
    audio_piece = audio_piece.reshape((1, AUDIO_PIECE_LENGTH, 1))  # Reshape the array such that the model can use it

    prediction = new_model.predict(audio_piece.reshape((1, AUDIO_PIECE_LENGTH, 1)))  # Predict the type of the audio and get scores for all categories
    print(prediction)  # TODO make better print statement that explains categories
