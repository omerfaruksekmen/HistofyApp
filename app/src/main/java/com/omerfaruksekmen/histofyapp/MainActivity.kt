package com.omerfaruksekmen.histofyapp

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private lateinit var tflite: Interpreter
    private lateinit var labels: List<String>

    private lateinit var imageView: ImageView
    private lateinit var txtResult: TextView

    private lateinit var txtInfo: TextView
    private lateinit var btnMap: Button
    private lateinit var btnSearch: Button

    private val IMAGE_SIZE = 224
    private val GALLERY_REQUEST = 1
    private val CAMERA_REQUEST = 2

    private val CAMERA_PERMISSION_CODE = 101
    private val STORAGE_PERMISSION_CODE = 102

    data class LandmarkInfo(
        val name: String,
        val description: String,
        val mapQuery: String
    )

    private val landmarkData = mapOf(
        "ayasofya" to LandmarkInfo(
            name = "Hagia Sophia",
            description = "Hagia Sophia is one of the most important structures in Istanbul, having witnessed both the Byzantine and Ottoman eras.",
            mapQuery = "Hagia Sophia Istanbul"
        ),
        "galata_kulesi" to LandmarkInfo(
            name = "Galata Tower",
            description = "The Galata Tower is a historic tower located in the Beyoğlu district of Istanbul and offers a unique view.",
            mapQuery = "Galata Tower"
        ),
        "kiz_kulesi" to LandmarkInfo(
            name = "Maidens Tower",
            description = "The Maiden's Tower is an iconic structure located in the Bosphorus Strait of Istanbul, renowned for its legends.",
            mapQuery = "Maidens Tower"
        )
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        imageView = findViewById(R.id.imageView)
        txtResult = findViewById(R.id.txtResult)

        txtInfo = findViewById(R.id.txtInfo)
        btnMap = findViewById(R.id.btnMap)
        btnSearch = findViewById(R.id.btnSearch)


        findViewById<Button>(R.id.btnGallery).setOnClickListener {
            checkStoragePermissionAndOpenGallery()
        }

        findViewById<Button>(R.id.btnCamera).setOnClickListener {
            checkCameraPermissionAndOpenCamera()
        }

        tflite = Interpreter(FileUtil.loadMappedFile(this, "mobilenetv2_model.tflite"))
        labels = FileUtil.loadLabels(this, "labels.txt")
    }

    private fun checkCameraPermissionAndOpenCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
        } else {
            openCamera()
        }
    }

    private fun checkStoragePermissionAndOpenGallery() {
        val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_IMAGES
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }

        if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(permission), STORAGE_PERMISSION_CODE)
        } else {
            openGallery()
        }
    }

    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(intent, CAMERA_REQUEST)
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(intent, GALLERY_REQUEST)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.isEmpty() || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Permission required", Toast.LENGTH_SHORT).show()
            return
        }

        when (requestCode) {
            CAMERA_PERMISSION_CODE -> openCamera()
            STORAGE_PERMISSION_CODE -> openGallery()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode != Activity.RESULT_OK || data == null) return

        val bitmap: Bitmap? = when (requestCode) {
            GALLERY_REQUEST -> {
                val uri: Uri? = data.data
                uri?.let { getBitmapFromUri(it) }
            }
            CAMERA_REQUEST -> {
                data.extras?.get("data") as? Bitmap
            }
            else -> null
        }

        bitmap?.let {
            imageView.setImageBitmap(it)
            classifyImage(it)
        }
    }

    private fun getBitmapFromUri(uri: Uri): Bitmap {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(contentResolver, uri)
            val bitmap = ImageDecoder.decodeBitmap(source)
            bitmap.copy(Bitmap.Config.ARGB_8888, true)
        } else {
            val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
            bitmap.copy(Bitmap.Config.ARGB_8888, true)
        }
    }

    private fun classifyImage(bitmap: Bitmap) {
        val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val input = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3)
        input.order(ByteOrder.nativeOrder())

        val intValues = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        resized.getPixels(intValues, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        for (pixel in intValues) {
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            input.putFloat(r)
            input.putFloat(g)
            input.putFloat(b)
        }

        val output = Array(1) { FloatArray(labels.size) }
        tflite.run(input, output)

        val prediction = output[0]
        val maxIdx = prediction.indices.maxByOrNull { prediction[it] } ?: -1
        val confidence = prediction[maxIdx]
        val confidenceThreshold = 0.6f

        if (confidence < confidenceThreshold || labels[maxIdx] == "diger") {
            txtResult.text = "This image could not be identified."

            txtInfo.visibility = View.GONE
            btnMap.visibility = View.GONE
            btnSearch.visibility = View.GONE
        } else {
            val result = labels[maxIdx]
            val confidencePercent = confidence * 100
            txtResult.text = "Prediction: $result"

            val labelKey = labels[maxIdx]

            if (landmarkData.containsKey(labelKey)) {
                val info = landmarkData[labelKey]!!

                txtInfo.text = info.description
                txtInfo.visibility = View.VISIBLE
                btnMap.visibility = View.VISIBLE
                btnSearch.visibility = View.VISIBLE

                btnMap.setOnClickListener {
                    val intent = Intent(Intent.ACTION_VIEW, Uri.parse("https://www.google.com/maps/search/?api=1&query=${Uri.encode(info.mapQuery)}"))
                    startActivity(intent)
                }

                btnSearch.setOnClickListener {
                    val intent = Intent(Intent.ACTION_VIEW, Uri.parse("https://www.google.com/search?q=${Uri.encode(info.name)}"))
                    startActivity(intent)
                }
            } else {
                txtInfo.visibility = View.GONE
                btnMap.visibility = View.GONE
                btnSearch.visibility = View.GONE
            }
        }
    }

}