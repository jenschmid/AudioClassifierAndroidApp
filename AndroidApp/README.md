# Android App
This project contains an Android App that can be used to classify microphone input from the smartphone 
into the classes "speech", "song" and "silence".

The structure of the app was taken and adapted from the 
[Tensorflow Sound Classification App](https://github.com/tensorflow/examples/tree/master/lite/examples/sound_classification/android).
Please refer to this project for further information about the App.

##General Comments
Since the task was to create a model that predicts the class of the last two seconds of the audio input from an 
Android App, we use the following parameter in the whole application:
* ```SAMPLE_RATE``` refers to the sample rate of the audio input and it is 16'000 or 16kHz since this is supported by 
  the microphone on the testing Android device
* ```AUDIO_PIECE_LENGTH``` rfers to the size of one audio input piece and it is 32'000 since we use a frequency of 
  16kHz and a sample length of 2 seconds
* ```N_MELS``` refers to the number of mels that are used for the creation of the mel spectrograms and it is 128
* Model related parameter such as batch size or number of training epochs are taken directly from the papers

## Installation and Run Instructions
The App was implemented in Android Studio.
The minimal requirements for the App are Android version 6.0 (Marshmallow) and SDK 23.

All requirements are handled in gradle.
Run a gradle sync to download all required libraries.

To start the app, use either an Android device emulator or an Android smartphone and start the App 
in [Android Studio](https://developer.android.com/studio).

## Tensorflow Lite Model
The tensorflow lite model used in this App was created and trained in the project 
[PythonModel](https://github.com/jeschm/AudioClassifierAndroidApp/tree/main/PythonModel) also contained in this GitHub repository.

Two different are available: 
* A 1D Convolutional Neuronal Network that works with raw audio input
* A 2D Convolutional Neuronal Network that works with mel spectrograms

To switch between the models please change the ``MODEL_FILE_PATH`` and the ``USE_MEL_SPEC`` flag 
in the ``MainActivity.kt`` class.

### Audio Input Transformation
Since one of the models requires a mel spectrogram of the audio signal as an input, the raw audio signal from the 
microphone must be transformed.
In the back end, the library [Librosa](https://librosa.org) is used for this purpose.
Since Librosa is a Python library and not available for Java or Kotlin, the 
[JLibrosa](https://github.com/Subtitle-Synchronizer/jlibrosa) library was used for the audio signal transformation.
This library offers the same transformations as the Librosa library and therefore allows to 
calculate mel spectrograms of the input audio in the same way as it is done in the back end for the model training.
The code from JLibrosa was adapted for this project and unused 
methods were removed.

## Resources and References
This list contains all resources that were used for the project:

* [TFLite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview)
* [JLibrosa](https://github.com/Subtitle-Synchronizer/jlibrosa)
* [Android Sound Classification App](https://github.com/tensorflow/examples/tree/master/lite/examples/sound_classification/android)
* [Guava](https://github.com/google/guava)
* [Apache Commons Mathematics Library](https://commons.apache.org/proper/commons-math/)



