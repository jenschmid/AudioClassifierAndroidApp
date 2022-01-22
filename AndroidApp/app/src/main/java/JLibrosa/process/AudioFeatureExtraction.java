package JLibrosa.process;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

/**
 * This Class calculates the MFCC, STFT values of given audio samples.
 *
 * Source based on https://github.com/chiachunfu/speech/blob/master/speechandroid/src/org/tensorflow/demo/mfcc/MFCC.java
 *
 * @author abhi-rawat1
 *
 */
public class AudioFeatureExtraction {

    private int n_mfcc = 40;
    private double sampleRate = 44100.0;


    private int length = -1;
    private double fMax = sampleRate / 2.0;
    private double fMin = 0.0;
    private int n_fft = 2048;
    private int hop_length = 512;
    private int n_mels = 128;






    /**
     * Variable for holding Sample Rate value
     *
     * @param sampleRateVal
     */
    public void setSampleRate(double sampleRateVal) {
        sampleRate = sampleRateVal;
        this.fMax = this.sampleRate/2.0;
    }


    public void setN_fft(int n_fft) {
        this.n_fft = n_fft;
    }

    public void setHop_length(int hop_length) {
        this.hop_length = hop_length;
    }

    public void setN_mels(int n_mels) {
        this.n_mels = n_mels;
    }







    /**
     * This function generates mel spectrogram values with extracted STFT features as complex values
     *
     * @param y
     * @return
     */
    public float [][] melSpectrogramWithComplexValueProcessing(float[] y) {

        Complex[][] spectro = extractSTFTFeaturesAsComplexValues(y, true);
        double[][] spectroAbsVal = new double[spectro.length][spectro[0].length];

        for(int i=0;i<spectro.length;i++) {
            for(int j=0;j<spectro[0].length;j++) {
                Complex complexVal = spectro[i][j];
                double spectroDblVal = Math.sqrt((Math.pow(complexVal.getReal(), 2) + Math.pow(complexVal.getImaginary(), 2)));
                spectroAbsVal[i][j] = Math.pow(spectroDblVal,2);
            }
        }

        double[][] melBasis = melFilter();
        float[][] melS = new float[melBasis.length][spectro[0].length];
        for (int i = 0; i < melBasis.length; i++) {
            for (int j = 0; j < spectro[0].length; j++) {
                for (int k = 0; k < melBasis[0].length; k++) {
                    melS[i][j] += melBasis[i][k] * spectroAbsVal[k][j];
                }
            }
        }
        return melS;


    }


    /**
     * This function extracts the STFT values as complex values
     *
     * @param y
     * @return
     */

    public Complex[][] extractSTFTFeaturesAsComplexValues(float[] y, boolean paddingFlag) {

        // Short-time Fourier transform (STFT)
        final double[] fftwin = getWindow();

        // pad y with reflect mode so it's centered. This reflect padding implementation
        // is
        final double[][] frame = padFrame(y, paddingFlag);
        double[][] fftmagSpec = new double[1 + n_fft / 2][frame[0].length];

        double[] fftFrame = new double[n_fft];

        Complex [][] complex2DArray = new Complex[1+n_fft/2][frame[0].length];
        Complex [] cmplx1DArr = new Complex [n_fft];


        float [][] invFrame = new float [n_fft][frame[0].length];

        for (int k = 0; k < frame[0].length; k++) {
            int fftFrameCounter = 0;
            for (int l = 0; l < n_fft; l++) {
                fftFrame[fftFrameCounter] = fftwin[l] * frame[l][k];
                fftFrameCounter = fftFrameCounter + 1;
            }

            double[] tempConversion = new double[fftFrame.length];
            double[] tempImag = new double[fftFrame.length];

            FastFourierTransformer transformer = new FastFourierTransformer(DftNormalization.STANDARD);

            try {
                Complex[] complx = transformer.transform(fftFrame, TransformType.FORWARD);

                Complex[] Invcomplx = transformer.transform(complx, TransformType.INVERSE);

                //FFT transformed data will be over the length of FFT
                //data will be sinusoidal in nature - so taking the values of 1+n_fft/2 only for processing
                for(int i=0;i<1+n_fft/2;i++) {
                    complex2DArray[i][k] = complx[i];

                }



                Complex [] cmplxINV1DArr = new Complex [n_fft];


                for(int j=0;j<1+n_fft/2;j++) {
                    cmplxINV1DArr[j] = complex2DArray[j][k];
                }

                int j_index = 2;
                for(int k1=1+n_fft/2;k1<n_fft;k1++){
                    cmplxINV1DArr[k1]=new Complex(cmplxINV1DArr[k1-j_index].getReal(), -1 * cmplxINV1DArr[k1-j_index].getImaginary());
                    j_index = j_index + 2;
                }

                Complex[] complx1 = transformer.transform(cmplxINV1DArr, TransformType.INVERSE);


                for(int p=0;p<complx1.length;p++) {
                    if(fftwin[p]!=0) {
                        invFrame[p][k] = (float) (complx1[p].getReal()/fftwin[p]);
                        //invFrame[p][i] = (float) (complx[p].getReal() * fftwin[p]);
                    }else {
                        invFrame[p][k] = 0;
                    }
                }


            } catch (IllegalArgumentException e) {
                System.out.println(e);
            }

        }


        float [] yValues = new float [(hop_length * (invFrame[0].length-1) + n_fft)];

        for (int i = 0; i < n_fft; i++) {
            for (int j = 0; j < invFrame[0].length; j++) {
                yValues[j*hop_length + i] = invFrame[i][j];

            }
        }


        return complex2DArray;

    }







    /**
     * This function pads the y values
     *
     * @param yValues
     * @return
     */

    private double[][] padFrame(float[] yValues, boolean paddingFlag){

        double[][] frame = null;

        if(paddingFlag) {


            double[] ypad = new double[n_fft + yValues.length];
            for (int i = 0; i < n_fft / 2; i++) {
                ypad[(n_fft / 2) - i - 1] = yValues[i + 1];
                ypad[(n_fft / 2) + yValues.length + i] = yValues[yValues.length - 2 - i];
            }
            for (int j = 0; j < yValues.length; j++) {
                ypad[(n_fft / 2) + j] = yValues[j];
            }

            frame = yFrame(ypad);
        }
        else {


            double[] yDblValues = new double[yValues.length];
            for (int i = 0 ; i < yValues.length; i++)
            {
                yDblValues[i] = (double) yValues[i];
            }

            frame = yFrame(yDblValues);

        }

        return frame;
    }


    /**
     * This function is used to get hann window, librosa
     *
     * @return
     */
    private double[] getWindow() {
        // Return a Hann window for even n_fft.
        // The Hann window is a taper formed by using a raised cosine or sine-squared
        // with ends that touch zero.
        double[] win = new double[n_fft];
        for (int i = 0; i < n_fft; i++) {
            win[i] = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / n_fft);
        }
        return win;
    }

    /**
     * This function is used to apply padding and return Frame
     *
     * @param ypad
     * @return
     */
    private double[][] yFrame(double[] ypad) {

        final int n_frames = 1 + (ypad.length - n_fft) / hop_length;

        double[][] winFrames = new double[n_fft][n_frames];

        for (int i = 0; i < n_fft; i++) {
            for (int j = 0; j < n_frames; j++) {
                winFrames[i][j] = ypad[j * hop_length + i];
            }
        }
        return winFrames;
    }




    /**
     * This function is used to create a Filterbank matrix to combine FFT bins into
     * Mel-frequency bins.
     *
     * @return
     */
    private double[][] melFilter() {
        // Create a Filterbank matrix to combine FFT bins into Mel-frequency bins.
        // Center freqs of each FFT bin
        final double[] fftFreqs = fftFreq();
        // 'Center freqs' of mel bands - uniformly spaced between limits
        final double[] melF = melFreq(n_mels + 2);

        double[] fdiff = new double[melF.length - 1];
        for (int i = 0; i < melF.length - 1; i++) {
            fdiff[i] = melF[i + 1] - melF[i];
        }

        double[][] ramps = new double[melF.length][fftFreqs.length];
        for (int i = 0; i < melF.length; i++) {
            for (int j = 0; j < fftFreqs.length; j++) {
                ramps[i][j] = melF[i] - fftFreqs[j];
            }
        }

        double[][] weights = new double[n_mels][1 + n_fft / 2];
        for (int i = 0; i < n_mels; i++) {
            for (int j = 0; j < fftFreqs.length; j++) {
                double lowerF = -ramps[i][j] / fdiff[i];
                double upperF = ramps[i + 2][j] / fdiff[i + 1];
                if (lowerF > upperF && upperF > 0) {
                    weights[i][j] = upperF;
                } else if (lowerF > upperF && upperF < 0) {
                    weights[i][j] = 0;
                } else if (lowerF < upperF && lowerF > 0) {
                    weights[i][j] = lowerF;
                } else if (lowerF < upperF && lowerF < 0) {
                    weights[i][j] = 0;
                } else {
                }
            }
        }

        double enorm[] = new double[n_mels];
        for (int i = 0; i < n_mels; i++) {
            enorm[i] = 2.0 / (melF[i + 2] - melF[i]);
            for (int j = 0; j < fftFreqs.length; j++) {
                weights[i][j] *= enorm[i];
            }
        }
        return weights;

        // need to check if there's an empty channel somewhere
    }

    /**
     * To get fft frequencies
     *
     * @return
     */
    private double[] fftFreq() {
        // Alternative implementation of np.fft.fftfreqs
        double[] freqs = new double[1 + n_fft / 2];
        for (int i = 0; i < 1 + n_fft / 2; i++) {
            freqs[i] = 0 + (sampleRate / 2) / (n_fft / 2) * i;
        }
        return freqs;
    }

    /**
     * To get mel frequencies
     *
     * @param numMels
     * @return
     */
    private double[] melFreq(int numMels) {
        // 'Center freqs' of mel bands - uniformly spaced between limits
        double[] LowFFreq = new double[1];
        double[] HighFFreq = new double[1];
        LowFFreq[0] = fMin;
        HighFFreq[0] = fMax;
        final double[] melFLow = freqToMel(LowFFreq);
        final double[] melFHigh = freqToMel(HighFFreq);
        double[] mels = new double[numMels];
        for (int i = 0; i < numMels; i++) {
            mels[i] = melFLow[0] + (melFHigh[0] - melFLow[0]) / (numMels - 1) * i;
        }
        return melToFreq(mels);
    }




    /**
     * To convert mel frequencies into hz frequencies
     *
     * @param mels
     * @return
     */
    private double[] melToFreq(double[] mels) {
        // Fill in the linear scale
        final double f_min = 0.0;
        final double f_sp = 200.0 / 3;
        double[] freqs = new double[mels.length];

        // And now the nonlinear scale
        final double min_log_hz = 1000.0; // beginning of log region (Hz)
        final double min_log_mel = (min_log_hz - f_min) / f_sp; // same (Mels)
        final double logstep = Math.log(6.4) / 27.0;

        for (int i = 0; i < mels.length; i++) {
            if (mels[i] < min_log_mel) {
                freqs[i] = f_min + f_sp * mels[i];
            } else {
                freqs[i] = min_log_hz * Math.exp(logstep * (mels[i] - min_log_mel));
            }
        }
        return freqs;
    }

    /**
     * To convert hz frequencies into mel frequencies
     *
     * @param freqs
     * @return
     */
    protected double[] freqToMel(double[] freqs) {
        final double f_min = 0.0;
        final double f_sp = 200.0 / 3;
        double[] mels = new double[freqs.length];

        // Fill in the log-scale part

        final double min_log_hz = 1000.0; // beginning of log region (Hz)
        final double min_log_mel = (min_log_hz - f_min) / f_sp; // # same (Mels)
        final double logstep = Math.log(6.4) / 27.0; // step size for log region

        for (int i = 0; i < freqs.length; i++) {
            if (freqs[i] < min_log_hz) {
                mels[i] = (freqs[i] - f_min) / f_sp;
            } else {
                mels[i] = min_log_mel + Math.log(freqs[i] / min_log_hz) / logstep;
            }
        }
        return mels;
    }

}
