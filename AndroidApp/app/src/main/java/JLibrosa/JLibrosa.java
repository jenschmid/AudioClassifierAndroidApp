package JLibrosa;

import JLibrosa.process.AudioFeatureExtraction;

/**
 * This Class is an equivalent of the Python Librosa utility used to extract the Audio features from given Wav file.
 * Code taken and adapted from: https://github.com/Subtitle-Synchronizer/jlibrosa
 * @author abhi-rawat1
 */
public class JLibrosa {
    public JLibrosa() {
    }

    /**
     * This function calculates and returns the mel spectrogram of a given audio sample
     *
     * @param yValues the input audio sample
     * @param mSampleRate the sample rate of the audio sample
     * @param n_fft the number of ffts that is used for the mel spectrogram calculation
     * @param n_mels the number of mels that is used for the mel spectrogram calculation
     * @param hop_length the hop length of the mel spectrogram calculation
     * @return The mel spectrogram of the given input audio
     */
    public float[][] generateMelSpectroGram(float[] yValues, int mSampleRate, int n_fft, int n_mels, int hop_length) {
        AudioFeatureExtraction mfccConvert = new AudioFeatureExtraction();
        mfccConvert.setSampleRate(mSampleRate);
        mfccConvert.setN_fft(n_fft);
        mfccConvert.setN_mels(n_mels);
        mfccConvert.setHop_length(hop_length);
        return mfccConvert.melSpectrogramWithComplexValueProcessing(yValues);
    }
}
