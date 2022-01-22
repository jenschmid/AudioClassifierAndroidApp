package JLibrosa;

import JLibrosa.process.AudioFeatureExtraction;


/**
 * This Class is an equivalent of Python Librosa utility used to extract the Audio features from given Wav file.
 *
 * @author abhi-rawat1
 */
public class JLibrosa {
    public JLibrosa() {
    }

    /**
     * This function calculates and returns the me of given Audio Sample
     * values. STFT stands for Short Term Fourier Transform
     */
    public float[][] generateMelSpectroGram(float[] yValues, int mSampleRate, int n_fft, int n_mels, int hop_length) {
        AudioFeatureExtraction mfccConvert = new AudioFeatureExtraction();
        mfccConvert.setSampleRate(mSampleRate);
        mfccConvert.setN_fft(n_fft);
        mfccConvert.setN_mels(n_mels);
        mfccConvert.setHop_length(hop_length);
        float[][] melSVal = mfccConvert.melSpectrogramWithComplexValueProcessing(yValues);
        return melSVal;
    }
}
