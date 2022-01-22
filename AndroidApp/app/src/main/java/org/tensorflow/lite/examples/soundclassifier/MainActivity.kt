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
import android.content.pm.PackageManager
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


class MainActivity : AppCompatActivity() {
    private val probabilitiesAdapter by lazy { ProbabilitiesAdapter() }
    private lateinit var handler: Handler
    private lateinit var jLibrosa: JLibrosa
    private var audioClassifier: AudioClassifier? = null
    private var audioRecord: AudioRecord? = null
    private var classificationInterval = 500L // how often should classification run in milli-secs

    var useMelSpectrogram = USE_MEL_SPEC

    private var SAMPLE_RATE = 16000

    private var nfft = 2048
    private var n_mels = 128
    private var hop_length = 512

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        with(binding) {
            recyclerView.apply {
                setHasFixedSize(false)
                adapter = probabilitiesAdapter
            }
        }

        // Create a handler to run classification in a background thread
        val handlerThread = HandlerThread("backgroundThread")
        handlerThread.start()
        handler = HandlerCompat.createAsync(handlerThread.looper)

        // Initialize the JLibrosa Object
        val librosa = JLibrosa()
        jLibrosa = librosa


        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestMicrophonePermission()
        } else {
            startAudioClassification()
        }
    }

    private fun startAudioClassification() {
        if (audioClassifier != null) return
        // If the audio classifier is initialized and running, do nothing.
        val classifier = AudioClassifier.createFromFile(this, MODEL_FILE)

        // Initialize the audio recorder
        val record = classifier.createAudioRecord()
        record.startRecording()

        // Define the classification runnable
        val run = object : Runnable {
            override fun run() {
                val output: List<Classifications>
                if (useMelSpectrogram) {
                val tensorCreated = TensorAudio.create(record.format, 32000)
                tensorCreated.load(record)

                    val spectogram2 = jLibrosa.generateMelSpectroGram(
                        tensorCreated.tensorBuffer.floatArray,
                        SAMPLE_RATE,
                        nfft,
                        n_mels,
                        hop_length
                    )

                    val flatArray: FloatArray = Floats.concat(*spectogram2)

                    val audioTensor2 = TensorAudio.create(record.format, flatArray.size)

                    audioTensor2.load(flatArray)

                    output = classifier.classify(audioTensor2)


                } else {
                    val tensorCreated = TensorAudio.create(record.format, 32000)
                    tensorCreated.load(record)
                    output = classifier.classify(tensorCreated)
                }

                // Filter out results above a certain threshold, and sort them descendingly
                val filteredModelOutput = output[0].categories.filter {
                    it.score > MINIMUM_DISPLAY_THRESHOLD
                }.sortedBy {
                    -it.score
                }


                // Updating the UI
                runOnUiThread {
                    probabilitiesAdapter.categoryList = filteredModelOutput
                    probabilitiesAdapter.notifyDataSetChanged()
                }

                // Rerun the classification after a certain interval
                handler.postDelayed(this, classificationInterval)
            }
        }

        // Start the classification process
        handler.post(run)
        audioClassifier = classifier
        audioRecord = record
    }

    private fun stopAudioClassification() {
        handler.removeCallbacksAndMessages(null)
        audioRecord?.stop()
        audioRecord = null
        audioClassifier = null
    }

    override fun onTopResumedActivityChanged(isTopResumedActivity: Boolean) {
        // Handles "top" resumed event on multi-window environment
        if (isTopResumedActivity) {
            startAudioClassification()
        } else {
            stopAudioClassification()
        }
    }

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


    override fun onDestroy() {
        super.onDestroy()
        println("OnDestroy")
        audioRecord!!.stop()
        audioRecord!!.release()
    }

    companion object {
        const val REQUEST_RECORD_AUDIO = 1337
        private const val TAG = "AudioDemo"
        private const val MODEL_FILE = "model_mel_metadata.tflite"
        private const val USE_MEL_SPEC = true
        private const val MINIMUM_DISPLAY_THRESHOLD: Float = 0.00000001f
    }
}
