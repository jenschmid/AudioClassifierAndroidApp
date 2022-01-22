"""
This script plots a sample signal and mel spectrogram as a visual plot.
The used signal is an array of length 32000 in a separate file and can be changed if wished.
"""

import matplotlib.pyplot as plot

from Constants import SAMPLE_RATE
from SampleSignal import sample_signal

plot.subplot(211)
plot.plot(sample_signal)
plot.xlabel('Sample')
plot.ylabel('Amplitude')

# Plot the spectrogram
plot.subplot(212)
plot.specgram(sample_signal, Fs=SAMPLE_RATE)
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.show()
