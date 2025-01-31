package com.example.yolov11

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.io.FileOutputStream
import java.util.*
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var rectView: RectView
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var session: OrtSession

    private val dataProcess = DataProcess(context = this)

    private var imageCapture: ImageCapture? = null

    companion object {
        const val PERMISSION_CAMERA = 1
        const val PERMISSION_STORAGE = 2
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        rectView = findViewById(R.id.rectView)

        // 자동 꺼짐 해제
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // 권한 허용
        setPermissions()

        // onnx 파일 && txt 파일 불러오기
        load()

        // 카메라 켜기
        setCamera()

        // 화면 캡처 버튼 (예시: 버튼을 클릭하면 캡처)
        val captureButton: View = findViewById(R.id.captureButton)
        captureButton.setOnClickListener {
            // 버튼 애니메이션 추가
            animateButton(captureButton)
            // 화면 캡처 기능 호출
            captureAndSaveScreen()
        }

    }

    private fun animateButton(button: View) {
        button.animate()
            .scaleX(0.9f) // 가로 크기 90%로 축소
            .scaleY(0.9f) // 세로 크기 90%로 축소
            .setDuration(100) // 축소 시간 100ms
            .withEndAction {
                button.animate()
                    .scaleX(1f) // 원래 크기로 복원
                    .scaleY(1f)
                    .setDuration(100) // 복원 시간 100ms
            }
    }

    private fun setCamera() {
        // 카메라 제공 객체
        val processCameraProvider = ProcessCameraProvider.getInstance(this).get()

        // 전체 화면
        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

        // 전면 카메라
        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

        // 16:9 화면으로 받아옴
        val preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .build()

        // preview 에서 받아와서 previewView 에 보여준다.
        preview.setSurfaceProvider(previewView.surfaceProvider)

        // ImageCapture 인스턴스 생성
        imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setTargetAspectRatio(AspectRatio.RATIO_16_9) // 여기에 비율 설정 추가
            .build()

        // 분석기 생성
        val analysis = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        analysis.setAnalyzer(Executors.newSingleThreadExecutor()) {
            imageProcess(it)
            it.close()
        }

        // 카메라의 수명 주기를 메인 액티비티에 귀속
        processCameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis, imageCapture)
    }


    private fun imageProcess(imageProxy: ImageProxy) {
        val bitmap = dataProcess.imageToBitmap(imageProxy)
        val floatBuffer = dataProcess.bitmapToFloatBuffer(bitmap)
        val inputName = session.inputNames.iterator().next() // session 이름
        val shape = longArrayOf(
            DataProcess.BATCH_SIZE.toLong(),
            DataProcess.PIXEL_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong()
        )
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuffer, shape)
        val resultTensor = session.run(Collections.singletonMap(inputName, inputTensor))
        val outputs = resultTensor.get(0).value as Array<*> // [1 84 8400]
        val results = dataProcess.outputsToNPMSPredictions(outputs)

        //화면 표출
        rectView.transformRect(results)
        rectView.invalidate()
    }

    private fun load() {
        dataProcess.loadModel() // onnx 모델 불러오기
        dataProcess.loadLabel() // coco txt 파일 불러오기

        ortEnvironment = OrtEnvironment.getEnvironment()
        session = ortEnvironment.createSession(
            this.filesDir.absolutePath.toString() + "/" + DataProcess.FILE_NAME,
            OrtSession.SessionOptions()
        )

        rectView.setClassLabel(dataProcess.classes)
    }

    private fun captureAndSaveScreen() {
        val imageCapture = imageCapture ?: return

        imageCapture.takePicture(
            Executors.newSingleThreadExecutor(),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    super.onCaptureSuccess(image)
                    val bitmap = imageProxyToBitmap(image)

                    // mutable bitmap 생성
                    val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)

                    val canvas = Canvas(mutableBitmap)

                    // ImageProxy의 원본 크기
                    val imageWidth = image.width
                    val imageHeight = image.height

                    // rectView의 크기
                    val rectViewWidth = rectView.width
                    val rectViewHeight = rectView.height

                    // 실제 비율 계산 (미리보기와 캡처된 이미지의 비율을 맞춤)
                    val scaleX = imageWidth.toFloat() / rectViewWidth.toFloat()
                    val scaleY = imageHeight.toFloat() / rectViewHeight.toFloat()

                    // 캔버스에 그리기 전에 스케일 적용
                    canvas.save()
                    canvas.scale(scaleX, scaleY)

                    // rectView의 내용을 mutableBitmap에 그리기
                    rectView.draw(canvas)

                    canvas.restore()

                    saveBitmapToGallery(mutableBitmap)  // 갤러리에 저장
                    image.close()
                }

                override fun onError(exception: ImageCaptureException) {
                    super.onError(exception)
                    Toast.makeText(this@MainActivity, "캡처 실패: ${exception.message}", Toast.LENGTH_SHORT).show()
                }
            })
    }



    // ImageProxy를 Bitmap으로 변환하는 함수
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)

        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    // Bitmap을 갤러리에 저장하는 함수
    private fun saveBitmapToGallery(bitmap: Bitmap) {
        try {
            val contentValues = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "screenshot_${System.currentTimeMillis()}.jpg")
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/${applicationContext.packageName}")  // 갤러리 폴더로 저장
            }

            // MediaStore를 통해 갤러리 저장
            val contentResolver = contentResolver
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

            uri?.let { uri ->
                val outputStream = contentResolver.openOutputStream(uri)
                outputStream?.use { stream ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream)
                    stream.flush()
                }

                // 갤러리에 저장이 완료되었을 때 Toast 표시 (UI 스레드에서 실행)
                runOnUiThread {
                    Toast.makeText(this, "이미지가 갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show()
                }
            } ?: run {
                // 실패할 경우 Toast 표시 (UI 스레드에서 실행)
                runOnUiThread {
                    Toast.makeText(this, "저장 실패", Toast.LENGTH_SHORT).show()
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
            runOnUiThread {
                Toast.makeText(this, "저장 실패", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        when (requestCode) {
            PERMISSION_CAMERA -> {
                if (grantResults.isEmpty() || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "카메라 권한이 거부되었습니다.", Toast.LENGTH_SHORT).show()
                    finish()
                }
            }
            PERMISSION_STORAGE -> {
                if (grantResults.isEmpty() || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "저장소 권한이 거부되었습니다.", Toast.LENGTH_SHORT).show()
                    finish()
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    private fun setPermissions() {
        // 카메라 권한 요청
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.CAMERA), PERMISSION_CAMERA)
        }

        // 저장소 권한 요청 (Android 10 이하에서 필요)
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.WRITE_EXTERNAL_STORAGE), PERMISSION_STORAGE)
            }
        }
    }
}
