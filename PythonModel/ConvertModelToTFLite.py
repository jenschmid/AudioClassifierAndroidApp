"""
Code from: https://www.tensorflow.org/lite/convert

This script converts a tensorflow model to a tensorflow lite model.
The lite version of the model is used by the Android app.
For more information check: https://www.tensorflow.org/lite/convert
"""
import tensorflow as tf

MODEL_PATH = "model/melspectrogram2/"
SAVE_TO_PATH = "model/melspectrogram2/converted/model_mel.tflite"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_PATH)  # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open(SAVE_TO_PATH, 'wb') as f:
    f.write(tflite_model)
