# Adnroid Audio Classifier
This project contains an Android App that classifies microphone input into speech, song, and silence.

The folder [PythonModel](https://github.com/jeschm/AudioClassifierAndroidApp/tree/main/PythonModel) contains all code and data used for the implementation of two tensorflow models.
Please refer to this folder for further information about the model specification, creation, training and the data.

The folder [AndroidApp](https://github.com/jeschm/AudioClassifierAndroidApp/tree/main/AndroidApp) contains the source code of the App that does the classification.
It uses the tensorflow lite model from the PythonModel folder.
Please refer to this folder for further information about the Android App.