package com.example.yolov11

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.round

class RectView(context: Context, attributeSet: AttributeSet) : View(context, attributeSet) {

    private var results: ArrayList<Result>? = null
    private lateinit var classes: Array<String>

    private val textPaint = Paint().also {
        it.textSize = 45f
        it.color = Color.WHITE
    }

    fun transformRect(results: ArrayList<Result>) {
        // scale 구하기
        val scaleX = width / DataProcess.INPUT_SIZE.toFloat()
        val scaleY = scaleX * 9f / 16f
        val realY = width * 9f / 16f
        val diffY = realY - height

        results.forEach {
            it.rectF.left *= scaleX
            it.rectF.right *= scaleX
            it.rectF.top = it.rectF.top * scaleY - (diffY / 2f)
            it.rectF.bottom = it.rectF.bottom * scaleY - (diffY / 2f)
        }
        this.results = results
    }

    override fun onDraw(canvas: Canvas?) {
        // 그림 그리기
        results?.forEach {
            // 사각형 그리기
            canvas?.drawRect(it.rectF, findPaint(it.classIndex))

            // 라벨 위치 조정 (박스 바깥 왼쪽 상단)
            val labelX = it.rectF.left + 10 // 왼쪽으로 약간 이동
            val labelY = it.rectF.top - 10 // 상단으로 약간 이동

            // 라벨 그리기
            canvas?.drawText(
                classes[it.classIndex] + ", " + round(it.score * 100) + "%",
                labelX,
                labelY,
                textPaint
            )
        }
        super.onDraw(canvas)
    }

    fun setClassLabel(classes: Array<String>) {
        this.classes = classes
    }

    //paint 지정
    private fun findPaint(classIndex: Int): Paint {
        val paint = Paint()
        paint.style = Paint.Style.STROKE    // 빈 사각형 그림
        paint.strokeWidth = 10.0f           // 굵기 10
        paint.strokeCap = Paint.Cap.ROUND   // 끝을 뭉특하게
        paint.strokeJoin = Paint.Join.ROUND // 끝 주위도 뭉특하게
        paint.strokeMiter = 100f            // 뭉특한 정도는 100도

        //임의로 지정한 색상
        paint.color = when (classIndex) {
            0 -> Color.WHITE
            else -> Color.DKGRAY
        }
        return paint
    }
}