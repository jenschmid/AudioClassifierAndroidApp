"""
This file contains code for testing the signal model with a single audio from a file in wav format.
"""
import tensorflow as tf
import librosa

from Constants import SAMPLE_RATE, cut_audio_pad_last, AUDIO_PIECE_LENGTH

audio_path = "data/data_short/song/en_low10.wav"  # The path to the audio sample that is loaded and used for testing
model_path = "model/signal/keras/"  # The path to the model that should be used

new_model = tf.keras.models.load_model(model_path)
audio, sample_rate_audio = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)

audio_pieces = cut_audio_pad_last(audio)  # Cut the audio sample into pieces of the right length

for audio_piece in audio_pieces:
    audio_piece = audio_piece.reshape((1, AUDIO_PIECE_LENGTH, 1))  # Reshape the array such that the model can use it

    prediction = new_model.predict(audio_piece.reshape((1, AUDIO_PIECE_LENGTH, 1)))  # Predict the class of the sample
    print("Predicted Probabilities")
    print("Silence: ", prediction[0][0])
    print("Song: ", prediction[0][1])
    print("Speech: ", prediction[0][2])
    print("--------------------")
