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
import android.view.WindowManager
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
    private var audioClassifierSpec: AudioClassifier? = null
    private var audioClassifierMel: AudioClassifier? = null
    private var audioClassifierUsed: AudioClassifier? = null
    private var audioRecord: AudioRecord? = null
    private var classificationInterval = 500L // how often should classification run in milli-secs

    var useMelSpectrogram = false

    private var AUDIO_PIECE_LENGTH = 32000
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

            // Input switch to turn on/off classification with mel spectrogram
            keepScreenOn(inputSwitch.isChecked)
            inputSwitch.setOnCheckedChangeListener { _, isChecked ->
                useMelSpectrogram = isChecked
                keepScreenOn(isChecked)
            }
        }

        // Create a handler to run classification in a background thread
        val handlerThread = HandlerThread("backgroundThread")
        handlerThread.start()
        handler = HandlerCompat.createAsync(handlerThread.looper)

        // Initialize the JLibrosa Object
        val librosa = JLibrosa()
        jLibrosa = librosa

        audioClassifierMel = AudioClassifier.createFromFile(this, MODEL_FILE_MEL)
        audioClassifierSpec = AudioClassifier.createFromFile(this, MODEL_FILE_SPEC)


        // Request microphone permission and start running classification
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestMicrophonePermission()
        } else {
            startAudioClassification()
        }
    }

    private fun startAudioClassification() {
        if (audioClassifierUsed != null) return;
        // If the audio classifier is initialized and running, do nothing.
        val classifier: AudioClassifier

        if (useMelSpectrogram) {
            classifier = audioClassifierMel!!
        } else {
            classifier = audioClassifierSpec!!
        }


        // Initialize the audio classifier
        val audioTensor = classifier.createInputTensorAudio()

        // Initialize the audio recorder
        val record = classifier.createAudioRecord()
        record.startRecording()

        // Define the classification runnable
        val run = object : Runnable {
            override fun run() {
                val startTime = System.currentTimeMillis()
                val output: List<Classifications>
                if (useMelSpectrogram) {
                    var bytesRead = 0
                    val floatInputBuffer = FloatArray(AUDIO_PIECE_LENGTH)
                    while (bytesRead < AUDIO_PIECE_LENGTH) {
                        audioTensor.load(record)
                        println("Short writing to file")
                        val remainingFloat =
                            audioTensor.tensorBuffer.buffer.asFloatBuffer().remaining()
                        if (AUDIO_PIECE_LENGTH - bytesRead > remainingFloat) {
                            audioTensor.tensorBuffer.buffer.asFloatBuffer()
                                .get(floatInputBuffer, bytesRead, remainingFloat)
                            bytesRead = bytesRead + remainingFloat
                        } else {
                            audioTensor.tensorBuffer.buffer.asFloatBuffer()
                                .get(floatInputBuffer, bytesRead, (AUDIO_PIECE_LENGTH - bytesRead))
                            bytesRead = bytesRead + (AUDIO_PIECE_LENGTH - bytesRead)
                        }
                    }
                    record.stop();


                    println("Creating spectrogram")
                    val spectogram2 = jLibrosa.generateMelSpectroGram(
                        floatInputBuffer,
                        SAMPLE_RATE,
                        nfft,
                        n_mels,
                        hop_length
                    )

                    val flatArray: FloatArray = Floats.concat(*spectogram2)

                    val audioTensor2 = TensorAudio.create(record.format, flatArray.size);

                    var bytesCopied = 0
                    val floatInputBufferOutput = FloatArray(flatArray.size)

                    audioTensor2.load(flatArray)

                    while (bytesCopied < flatArray.size) {

                        val remaining = flatArray.size - bytesCopied
                        if (remaining < 4000) {
                            audioTensor2.load(
                                flatArray.copyOfRange(bytesCopied, remaining),
                                bytesCopied,
                                remaining
                            )
                            bytesCopied = bytesCopied + remaining
                        } else {
                            audioTensor2.load(
                                flatArray.copyOfRange(
                                    bytesCopied,
                                    bytesCopied + 4000
                                ), bytesCopied, 4000
                            )
                            bytesCopied = bytesCopied + 4000
                        }

                    }
                    output = classifier.classify(audioTensor2)


                } else {
                    val tensorCreated = TensorAudio.create(record.format, 32000);
                    tensorCreated.load(record)


//                    var bytesRead = 0
//                    val floatInputBuffer = FloatArray(songPieceSize)
//                    while (bytesRead < songPieceSize) {
//                        audioTensor.load(record)
//                        println("Short writing to file")
//                        val remainingFloat = songPieceSize - bytesRead
//                        if (songPieceSize - bytesRead > remainingFloat) {
//                            audioTensor.tensorBuffer.buffer.asFloatBuffer()
//                                .get(floatInputBuffer, bytesRead, remainingFloat)
//                            tensorCreated.load(floatInputBuffer)
//                            bytesRead = bytesRead + remainingFloat
//                        } else {
//                          val readSize = 4000
//                            audioTensor.tensorBuffer.buffer.asFloatBuffer()
//                                .get(floatInputBuffer, bytesRead, readSize)
//                            tensorCreated.load(floatInputBuffer, bytesRead, readSize)
//                            bytesRead = bytesRead + readSize
//                        }
//                    }
                    //record.stop();
                    output = classifier.classify(tensorCreated)
                }

                // Filter out results above a certain threshold, and sort them descendingly
                val filteredModelOutput = output[0].categories.filter {
                    it.score > MINIMUM_DISPLAY_THRESHOLD
                }.sortedBy {
                    -it.score
                }

                val finishTime = System.currentTimeMillis()

                Log.d(TAG, "Latency = ${finishTime - startTime}ms")

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

        audioClassifierUsed = classifier
        audioRecord = record
    }

    private fun stopAudioClassification() {
        handler.removeCallbacksAndMessages(null)
        audioRecord?.stop()
        audioRecord = null
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

    private fun keepScreenOn(enable: Boolean) =
        if (enable) {
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } else {
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }

    override fun onDestroy() {
        super.onDestroy()
        println("OnDestroy")
        audioRecord!!.release()
    }

    companion object {
        const val REQUEST_RECORD_AUDIO = 1337
        private const val TAG = "AudioDemo"
        private const val MODEL_FILE_SPEC = "model_spec_metadata.tflite"
        private const val MODEL_FILE_MEL = "model_mel_metadata.tflite"
        private const val MINIMUM_DISPLAY_THRESHOLD: Float = 0.00000001f
    }
}
