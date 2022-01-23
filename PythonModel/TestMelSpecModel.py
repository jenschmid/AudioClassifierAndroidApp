"""
This file contains code for testing the mel spectrogram model with a single audio from a file in wav format.
"""
import tensorflow as tf
import librosa

from Constants import cut_audio_pad_last, SAMPLE_RATE, N_MELS, IS_MONO

audio_path = "data/data_short/speech/speech_low10.wav"  # The path to the audio that is loaded and used for testing
model_path = "model/melspectrogram/keras/"  # The path to the model that should be used

new_model = tf.keras.models.load_model(model_path)
audio, sample_rate_audio = librosa.load(audio_path, sr=SAMPLE_RATE, mono=IS_MONO)

audio_pieces = cut_audio_pad_last(audio)  # Cut the audio sample into pieces of the right length

for audio_piece in audio_pieces:
    melspect = librosa.feature.melspectrogram(y=audio_piece, sr=SAMPLE_RATE, n_mels=N_MELS)  # Compute a mel spectrogram
    melspect = melspect.reshape((1, N_MELS, 63))  # Reshape the array such that the model can use it

    prediction = new_model.predict(melspect)  # Predict the class of the sample
    print("Predicted Probabilities")
    print("Silence: ", prediction[0][0])
    print("Song: ", prediction[0][1])
    print("Speech: ", prediction[0][2])
    print("--------------------")
