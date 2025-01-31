package com.example.yolov11

import android.graphics.RectF

data class Result(val classIndex: Int, val score: Float, val rectF: RectF)
