# Python Backend for Audio Classification
This folder contains the Python backend that was used for the implementation and training of two different models.
License, Author etc.


## Installation and Run Instructions
All requirements are stored in the requirements.txt file. 
In order to install them, please run the command ```pip install```.

To create the models, that can be used by the Android App, please run the following scripts:

1. ```TrainModel.py```
2. ```ConvertModelToTFLite.py```
3. ```WriteMetadataToModel.py```

After running all three scripts, copy the tensorflow lite model that contains the metadata to your Android App.

## Script Explanation
This section gives a short introduction to each script.

### Constants.py

### labels.txt

### TrainModel.py

### ConvertModelToTFLite.py

### WriteMetadataToModel.py

### PlotSpectrogram.py

### SampleSpectrogram.py

### TestSpecModel.py

### TestMelSpecModel.py

## References and Resources

* [`Tensorflow Metadata Writer Information`](https://www.tensorflow.org/lite/convert/metadata)
* [`Tensorflow Metadata Converter (Code)`](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/convert/metadata_writer_tutorial.ipynb)
* [`Tensorflow Lite Model Converter`](https://www.tensorflow.org/lite/convert)
* [`1D CNN Model for Spectrogram Data`](https://github.com/Logan97117/environmental_sound_classification_1DCNN)

