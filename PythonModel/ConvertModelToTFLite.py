"""
Code from: https://www.tensorflow.org/lite/convert

This script converts a tensorflow model to a tensorflow lite model.
The lite version of the model is used by the Android app.
For more information check: https://www.tensorflow.org/lite/convert
"""
import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('model/spectrogram/')  # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model/spectrogram/converted/model_spec.tflite', 'wb') as f:
    f.write(tflite_model)
