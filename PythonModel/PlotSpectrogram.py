"""
This script plots a sample spectrogram and melspectrogram as a visual plot.
The used spectrogram is an array of length 32000 in a separate file and can be changed if wished.
"""

import matplotlib.pyplot as plot

from Constants import SAMPLE_RATE
from SampleSpectrogram import sample_spectrogram

plot.subplot(211)
plot.plot(sample_spectrogram)
plot.xlabel('Sample')
plot.ylabel('Amplitude')

# Plot the spectrogram
plot.subplot(212)
plot.specgram(sample_spectrogram, Fs=SAMPLE_RATE)
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.show()
