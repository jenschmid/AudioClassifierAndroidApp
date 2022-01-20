# Android App
This project contains an Android App that can be used to classify microphone input into the classes "speech", "singing" and "silence".

The structure of the app was taken and adapted from:
Please refer to this project for further information about the App structure, the run instructions and requirements.


## Tensorflow Lite Model
The tensorflow lite model used in this App were created and trained in the project PythonModel also contained in this GitHub repository.
Two different models were used: A 1D Convolutional neuronal network that works with raw audio input, and a 2D convolutional neural network that works with mel spectrograms.
The app allows to switch between the two models while running.
For further information abotu the models, please check the PythonModel project.

## Adaptions to original code
In addition to the models that were exchanged, the following adaptions have been made to the App.

### Mel Spectrogram Creation
For the 2D model, a 

### Presentation of all Classes and Probabilities
In the original App, only labels with a minimum probability were shown in the UI.
in the adapted code, all labels are shown (even if their probability is zero).
This was chosen to increase the debugging of the App and to better validate the classification output.


### Sample rate, sample length and classification interval
The sample rate and sample lenth are fixed in this model such that they match the parameters that were used to train the model.
The sample rate is set to 16000 Hz  and the sample length is set to 320000.

The classification interval is set to 100 miliseconds, meaning that the classification is run every 100 miliseconds.
In the original code, the classification interval was variable and could be adapted in the UI.





## Resources and References

JLibrosa code

This Android application demonstrates how to classify sound on-device. It uses:

* [TFLite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview)
* [YAMNet](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1),
an audio event classification model.

## Requirements

*   Android Studio 4.1 (installed on a Linux, Mac or Windows machine)
*   An Android device with Android 6.0+



