package com.example.camera2apinew

import android.net.Uri
import androidx.lifecycle.ViewModel


/**
    It's basically inside the class is where you want to set your values that
    you want to share between the other fragments
 */
// TODO
class CameraViewModel : ViewModel() {
    var videoUri: Uri? = null
}