/*
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.soundclassifier

import JLibrosa.JLibrosa
import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.AudioRecord
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.os.HandlerCompat
import com.google.common.primitives.Floats
import org.tensorflow.lite.examples.soundclassifier.databinding.ActivityMainBinding
import org.tensorflow.lite.support.audio.TensorAudio
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import org.tensorflow.lite.task.audio.classifier.Classifications

/**
 * This Class creates the main entry point for the Android Audio Classification App
 * The code was taken and adapted from: https://github.com/tensorflow/examples/tree/master/lite/examples/sound_classification/android
 */
class MainActivity : AppCompatActivity() {
    private val probabilitiesAdapter by lazy { ProbabilitiesAdapter() }
    private lateinit var handler: Handler
    private lateinit var jLibrosa: JLibrosa
    private var audioClassifier: AudioClassifier? = null
    private var audioRecord: AudioRecord? = null

    private var CLASSIFICATION_INTERVAL = 500L //How often should classification run in milli-secs

    // The following parameters must have the same value as the ones that were used
    // for the model creation. For more information about the values, please check the README
    private var SAMPLE_RATE = 16000 //The sample rate of the input audio
    private var NUMBER_OF_FFT = 2048 //Number of ffts that is used for the mel spectrogram
    private var NUMBER_OF_MELS = 128 //The number of mels that is used for the mel spectrogram
    private var HOP_LENGTH = 512 //The hop length that is used for the mel spectrogram

    var USE_MELSPECTROGRAM =
        USE_MEL_SPEC //Indicates whether mel spectrograms should be used for the classification

    /**
     * This method handles all start up events that must be done before the classification can run
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Inflate the layout
        val binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Add the probabilities adapter
        with(binding) {
            recyclerView.apply {
                setHasFixedSize(false)
                adapter = probabilitiesAdapter
            }
        }

        // Change the audio mode to COMMUNICATION since this makes the microphone input better
        val am: AudioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        am.mode = AudioManager.MODE_IN_COMMUNICATION

        // Create a handler to run classification in a background thread
        val handlerThread = HandlerThread("backgroundThread")
        handlerThread.start()
        handler = HandlerCompat.createAsync(handlerThread.looper)

        // Initialize the JLibrosa Object
        val librosa = JLibrosa()
        jLibrosa = librosa

        // Check whether the app must ask the user for permission to use the microphone input
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestMicrophonePermission()
        } else {
            startAudioClassification()
        }
    }

    /**
     * This method runs the classification of audio input samples
     */
    private fun startAudioClassification() {
        // If the audio classifier is initialized and running, do nothing
        if (audioClassifier != null) return

        // Else, load the tensorflow lite classification model from a file
        val classifier = AudioClassifier.createFromFile(this, MODEL_FILE_PATH)

        // Initialize the audio recorder and start recording
        val record = classifier.createAudioRecord()
        record.startRecording()

        // Define the classification runnable
        val run = object : Runnable {
            override fun run() {
                // Define the result list
                val result: List<Classifications>

                // Load the microphone input
                val inputTensor = TensorAudio.create(record.format, 32000)
                inputTensor.load(record)

                // Check if the classification method should use mel spectrograms
                if (USE_MELSPECTROGRAM) {

                    // If yes, mel spectrograms need to be calculated with the JLibrosa library
                    val spectogram2 = jLibrosa.generateMelSpectroGram(
                        inputTensor.tensorBuffer.floatArray,
                        SAMPLE_RATE,
                        NUMBER_OF_FFT,
                        NUMBER_OF_MELS,
                        HOP_LENGTH
                    )

                    // The calculated mel spectrogram must be reshaped into the right form
                    val flatArray: FloatArray = Floats.concat(*spectogram2)
                    val spectrogramTensor = TensorAudio.create(record.format, flatArray.size)
                    spectrogramTensor.load(flatArray)
                    result = classifier.classify(spectrogramTensor)
                } else {
                    // If mel spectrograms are not used, use the raw input signal for classification
                    result = classifier.classify(inputTensor)
                }

                // Filter out results above a certain threshold, and sort them descendingly
                val filteredModelOutput = result[0].categories.filter {
                    it.score > MINIMUM_DISPLAY_THRESHOLD
                }.sortedBy {
                    -it.score
                }

                // Update the UI with the probabilities from the classification
                runOnUiThread {
                    probabilitiesAdapter.categoryList = filteredModelOutput
                    probabilitiesAdapter.notifyDataSetChanged()
                }

                // Rerun the classification after a certain interval
                handler.postDelayed(this, CLASSIFICATION_INTERVAL)
            }
        }

        // Start the classification process
        handler.post(run)
        audioClassifier = classifier
        audioRecord = record
    }

    /**
     * This method stops the audio classification and removes the audio record and classifier from the class
     */
    private fun stopAudioClassification() {
        handler.removeCallbacksAndMessages(null)
        audioRecord?.stop()
        audioRecord = null
        audioClassifier = null
    }

    /**
     * This method handles "top" resumed event on multi-window environment
     * @param isTopResumedActivity indicates whether the action is a top resumed activity
     */
    override fun onTopResumedActivityChanged(isTopResumedActivity: Boolean) {
        if (isTopResumedActivity) {
            startAudioClassification()
        } else {
            stopAudioClassification()
        }
    }

    /**
     * This method handles the result of the permission check
     */
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode == REQUEST_RECORD_AUDIO) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.i(TAG, "Audio permission granted :)")
                startAudioClassification()
            } else {
                Log.e(TAG, "Audio permission not granted :(")
            }
        }
    }

    /**
     * This method checks if the app must ask the user for permission to use the microphone.
     * If the permission must be requested, the method handles the request.
     */
    @RequiresApi(Build.VERSION_CODES.M)
    private fun requestMicrophonePermission() {
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startAudioClassification()
        } else {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
        }
    }


    /**
     * Handles the onDestroy event (when the App is closed).
     * The method releases all resources and changes the AudioMode back to normal
     */
    override fun onDestroy() {
        super.onDestroy()
        //Release the AudioRecord resource
        audioRecord!!.stop()
        audioRecord!!.release()

        //Change the Audio Mode back to normal
        val am: AudioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        am.mode = AudioManager.MODE_NORMAL
    }

    companion object {
        const val REQUEST_RECORD_AUDIO = 1337
        private const val TAG = "AudioDemo"

        private const val MODEL_FILE_PATH = "model_mel_metadata.tflite" // Path to the model
        private const val USE_MEL_SPEC = true // This should be set to true if the loaded model works with mel spectrograms

        private const val MINIMUM_DISPLAY_THRESHOLD: Float = 0.00000001f
    }
}
