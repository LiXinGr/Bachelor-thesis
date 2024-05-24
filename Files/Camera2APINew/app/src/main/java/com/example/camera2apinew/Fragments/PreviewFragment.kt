package com.example.camera2apinew.Fragments

import android.Manifest
import android.R.attr.height
import android.R.attr.width
import android.content.Context
import android.content.res.Configuration
import android.graphics.ImageFormat
import android.graphics.Point
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.StreamConfigurationMap
import android.media.Image
import android.media.ImageReader
import android.media.MediaRecorder
import android.media.MediaScannerConnection
import android.media.ThumbnailUtils
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import android.util.Range
import android.util.Size
import android.util.SparseIntArray
import android.view.LayoutInflater
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.CheckBox
import android.widget.Chronometer
import android.widget.ImageView
import android.widget.SeekBar
import android.widget.TextView
import androidx.core.graphics.drawable.RoundedBitmapDrawable
import androidx.core.graphics.drawable.RoundedBitmapDrawableFactory
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.camera2apinew.CameraViewModel
import com.example.camera2apinew.R
import com.vmadalin.easypermissions.EasyPermissions
import com.vmadalin.easypermissions.annotations.AfterPermissionGranted
import java.io.File
import java.text.SimpleDateFormat
import java.util.Arrays
import java.util.Collections
import java.util.Date
import java.util.Timer
import java.util.TimerTask


class PreviewFragment : Fragment() {


    /**
     * Orientation of the camera sensor
     */
    private var mSensorOrientation = 0
    val desiredAspectRatio = 16.0 / 9.0
    lateinit var previewTextureView : TextureView
    var previewControlBox : CheckBox? = null
    var captureButton : ImageView? = null
    var thumbButton : ImageView? = null
    var chronometer: Chronometer? = null
    var isoSeekBar: SeekBar? = null
    var isoSeekBarValueText: TextView? = null
    var isoValueText: TextView? = null
    var isoValueProg = 0
    // Nanoseconds
    var exposureTimeValueProg : Long = 0
    var exposureTimeSeekBar: SeekBar? = null
    var exposureTimeSeekBarValueText: TextView? = null
    var exposureTimeValueText: TextView? = null
    var focusValueText: TextView? = null
    var focusSeekBar: SeekBar? = null
    var focusSeekBarValueText: TextView? = null
    var focusValueProg = 0.0f
    var previewControlBool = false
    var frameRateValueText: TextView? = null
    // Nanoseconds
    var frameRateValueProg = 30
    var frameRateSeekBar: SeekBar? = null
    var frameRateSeekBarValueText: TextView? = null
    //var apertureAvailableRangeList : List<T> = null
    val cameraManager by lazy {
        activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }
    //////
    private lateinit var handler: Handler
    private lateinit var timer: Timer

    private val mediaRecorder by lazy {
        MediaRecorder()
    }
    private lateinit var currentVideoFilePath : String

    private lateinit var backgroundThread: HandlerThread
    private lateinit var backgroundHandler: Handler

    private val MAX_PREVIEW_WIDTH = 1920
    private val MAX_PREVIEW_HEIGHT = 1080
    private lateinit var captureSession: CameraCaptureSession
    private lateinit var captureRequestBuilder: CaptureRequest.Builder
    private lateinit var cameraId: String
    private var isRecording = false


    // TODO
    private val cameraViewModel by lazy {
        ViewModelProvider(activity!!).get(CameraViewModel::class.java)
    }

    // TODO
    /**
     * The [android.util.Size] of camera preview.
     */
    //private var mPreviewSize: Size? = null

    private lateinit var cameraDevice: CameraDevice

    // TODO
    companion object {
        const val REQUEST_CAMERA_AND_STORAGE_PERMISSION = 100
        private val TAG = PreviewFragment::class.qualifiedName
        @JvmStatic fun newInstance() = PreviewFragment()

        private const val SENSOR_0_ORIENTATION_DEGREES = 0
        private const val SENSOR_90_ORIENTATION_DEGREES = 90
        private const val SENSOR_180_ORIENTATION_DEGREES = 180
        private const val SENSOR_270_ORIENTATION_DEGREES = 270
        private val ORIENTATION_0 = SparseIntArray().apply {
            append(Surface.ROTATION_0, 0)
            append(Surface.ROTATION_90, 270)
            append(Surface.ROTATION_180, 180)
            append(Surface.ROTATION_270, 90)
        }
        private val ORIENTATION_90 = SparseIntArray().apply {
            append(Surface.ROTATION_0, 90)
            append(Surface.ROTATION_90, 0)
            append(Surface.ROTATION_180, 270)
            append(Surface.ROTATION_270, 180)
        }
        private val ORIENTATION_180 = SparseIntArray().apply {
            append(Surface.ROTATION_0, 180)
            append(Surface.ROTATION_90, 90)
            append(Surface.ROTATION_180, 0)
            append(Surface.ROTATION_270, 270)
        }
        private val ORIENTATION_270 = SparseIntArray().apply {
            append(Surface.ROTATION_0, 270)
            append(Surface.ROTATION_90, 180)
            append(Surface.ROTATION_180, 90)
            append(Surface.ROTATION_270, 0)
        }
    }

    ////////////////////////////////////// Automatically called methods //////////////////////////////////////////////////


    /**
     Creating links to buttons
     */
    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        val rootView = inflater.inflate(R.layout.fragment_preview, container, false)

        previewTextureView = rootView.findViewById(R.id.previewTextureView)
        thumbButton = rootView.findViewById(R.id.thumbButton)
        captureButton = rootView.findViewById(R.id.captureButton)
        chronometer = rootView.findViewById(R.id.chronometer2)
        previewControlBox = rootView.findViewById(R.id.checkBox)
        // ISO
        isoSeekBar = rootView.findViewById(R.id.isoSeekBar)
        isoSeekBarValueText = rootView.findViewById(R.id.isoSeekBarValueText)
        isoValueProg = isoSeekBar!!.progress
        isoValueText = rootView.findViewById(R.id.isoCurrentValue)
        // Exposure Time
        exposureTimeSeekBar = rootView.findViewById(R.id.exposureTimeSeekBar)
        exposureTimeSeekBarValueText = rootView.findViewById(R.id.exposureTimeSeekBarValueText)
        exposureTimeValueProg = seekBarExposureTimeProgressToProgValue(exposureTimeSeekBar!!.progress)
        exposureTimeSeekBar!!.progress
        exposureTimeValueText = rootView.findViewById(R.id.exposureTimeValueText)
        // Focus
        focusValueText = rootView.findViewById(R.id.focusValueText)
        focusSeekBar = rootView.findViewById(R.id.focusSeekBar)
        focusSeekBarValueText = rootView.findViewById(R.id.focusSeekBarValueText)
        focusValueProg = focusSeekBar!!.progress/100f // TODO: do the function
        // Frame Rate
        frameRateValueText = rootView.findViewById(R.id.frameRateValueText)
        frameRateSeekBar = rootView.findViewById(R.id.frameRateSeekBar)
        frameRateSeekBarValueText = rootView.findViewById(R.id.frameRateSeekBarValueText)
        // TODO constistency
        //frameRateValueProg = frameRateSeekBar!!.progress
        handler = Handler(Looper.getMainLooper())
        timer = Timer()

        return rootView
    }

    /**
    In summary, the code within onViewCreated() sets specific flags for the system UI
    to create an immersive experience for the user,
    where the status bar and navigation bar are hidden, and the content layout adjusts appropriately.
     **/
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        activity?.window?.decorView?.systemUiVisibility = View.SYSTEM_UI_FLAG_FULLSCREEN or
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE
    }

    /**
     Button listeners
     */
    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)

        captureButton!!.setOnClickListener{
            Log.d(TAG, "capture button selected")
            if (isRecording){
                isRecording = false
                stopRecordSession()
            } else {
                setupMediaRecorder()
                //isRecording = true
                startRecordSession()
            }
        }

        thumbButton!!.setOnClickListener{
            Log.d(TAG, "thumbnail button selected")
            val exoPlayerFragment = ExoPlayerFragment.newInstance()
            replaceFragment(exoPlayerFragment)
        }

        frameRateSeekBar!!.setOnSeekBarChangeListener(object: SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(p0: SeekBar?, progress: Int, p2: Boolean) {
                frameRateValueProg = progress
                // TODO: do i really need it here???
                frameRateSeekBarValueText!!.text = "fps: $progress}"
                captureRequestBuilder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(frameRateValueProg, frameRateValueProg))
                setRepeatingRequest()
            }

            override fun onStartTrackingTouch(p0: SeekBar?) {
                // TODO: do i really need it here???
                frameRateValueProg = frameRateSeekBar!!.progress
                // TODO: do i really need it here???
                captureRequestBuilder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(frameRateValueProg, frameRateValueProg))
            }

            override fun onStopTrackingTouch(p0: SeekBar?) {
                // TODO: do i really need it here???
                frameRateValueProg = frameRateSeekBar!!.progress
                // TODO: do i really need it here???
                captureRequestBuilder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(frameRateValueProg, frameRateValueProg))
            }

        })

        isoSeekBar!!.setOnSeekBarChangeListener(object: SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(p0: SeekBar?, p1: Int, p2: Boolean) {
                isoValueProg = isoSeekBar!!.progress
                // TODO: do i really need it here???
                isoSeekBarValueText!!.text = "ISO: ${isoSeekBar!!.progress}"
                captureRequestBuilder.set(CaptureRequest.SENSOR_SENSITIVITY, isoValueProg)
                setRepeatingRequest()


                val iso = captureRequestBuilder.get(CaptureRequest.SENSOR_SENSITIVITY)
                Log.d("CameraSettings", "ISO: $iso")

                // Query shutter speed
                val shutterSpeed = captureRequestBuilder.get(CaptureRequest.SENSOR_EXPOSURE_TIME)
                Log.d("CameraSettings", "ET: $shutterSpeed")

                // Query white balance mode
                val wbMode = captureRequestBuilder.get(CaptureRequest.CONTROL_AWB_MODE)
                Log.d("CameraSettings", "White Balance Mode: $wbMode")

                // Query focus distance
                val focusDistance = captureRequestBuilder.get(CaptureRequest.LENS_FOCUS_DISTANCE)
                Log.d("CameraSettings", "Focus Distance: $focusDistance")
            }

            override fun onStartTrackingTouch(p0: SeekBar?) {
                // TODO: do i really need it here???
                isoValueProg = isoSeekBar!!.progress
                captureRequestBuilder.set(CaptureRequest.SENSOR_SENSITIVITY, isoValueProg)
                setRepeatingRequest()
            }

            override fun onStopTrackingTouch(p0: SeekBar?) {
                // TODO: do i really need it here???
                isoValueProg = isoSeekBar!!.progress
                captureRequestBuilder.set(CaptureRequest.SENSOR_SENSITIVITY, isoValueProg)
                setRepeatingRequest()
                //Log.d(TAG, "ISO value: $isoValueProg")
            }

        })

        exposureTimeSeekBar!!.setOnSeekBarChangeListener(object: SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(p0: SeekBar?, progress: Int, p2: Boolean) {
                exposureTimeValueProg = seekBarExposureTimeProgressToProgValue(progress)
                // TODO: do i really need it here???
                exposureTimeSeekBarValueText!!.text = "ET: 1/${exposureTimeSeekBar!!.progress}"
                captureRequestBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureTimeValueProg)
                setRepeatingRequest()
            }

            override fun onStartTrackingTouch(p0: SeekBar?) {
                // TODO: do i really need it here???
                exposureTimeValueProg = seekBarExposureTimeProgressToProgValue(exposureTimeSeekBar!!.progress)
                captureRequestBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureTimeValueProg)
                //setRepeatingRequest()
            }

            override fun onStopTrackingTouch(p0: SeekBar?) {
                // TODO: do i really need it here???
                exposureTimeValueProg = seekBarExposureTimeProgressToProgValue(exposureTimeSeekBar!!.progress)
                captureRequestBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureTimeValueProg)
                //setRepeatingRequest()
            }

        })

        focusSeekBar!!.setOnSeekBarChangeListener(object: SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(p0: SeekBar?, progress: Int, p2: Boolean) {
                focusValueProg = focusSeekBar!!.progress/100f
                // TODO: do i really need it here???
                focusSeekBarValueText!!.text = "Focus: ${focusValueProg}"
                captureRequestBuilder.set(CaptureRequest.LENS_FOCUS_DISTANCE, focusValueProg)
                setRepeatingRequest()
            }

            override fun onStartTrackingTouch(p0: SeekBar?) {
                // TODO: do i really need it here???
                focusValueProg = focusSeekBar!!.progress/100f
                captureRequestBuilder.set(CaptureRequest.LENS_FOCUS_DISTANCE, focusValueProg)
                //setRepeatingRequest()
            }

            override fun onStopTrackingTouch(p0: SeekBar?) {
                // TODO: do i really need it here???
                focusValueProg = focusSeekBar!!.progress/100f
                captureRequestBuilder.set(CaptureRequest.LENS_FOCUS_DISTANCE, focusValueProg)
                //setRepeatingRequest()
            }

        })

        previewControlBox!!.setOnCheckedChangeListener{ buttonView, isCheck ->
            previewControlBool = isCheck
            if (isCheck){
                setProgressBarToManualControl()
            }
            if(!isRecording)
                previewSession()
            else
                recordSession()
        }

    }

    /**
    Method is called when the activity comes into the foreground
    and becomes visible to the user.
    Resume any resources, connections, or services
    that were released or paused in the onPause() or onStop() methods.
    You can use onResume() to update the UI elements, refresh content,
    or handle any changes that occurred while the activity was paused or stopped.
     */
    override fun onResume() {
        super.onResume()

        startBackgroundThread()
        // TODO: check if camera was already opened
        if (previewTextureView.isAvailable) {
            // If the TextureView is already available, open the camera immediately
            Log.i(TAG, "the TextureView is already available")
            openCamera()
        } else {
            // If the TextureView is not yet available, set up the SurfaceTextureListener
            Log.i(TAG, "the TextureView is NOT yet available")
            previewTextureView.surfaceTextureListener = surfaceListener
        }
    }

    /**
    Method is called when the activity is about to go into the background
    and lose focus but is still visible to the user.
    Pause ongoing operations, release resources, or stop services
    that are not needed when the activity is not in the foreground
    to optimize performance and battery usage.
    Save the activity's current state, data, or user input,
    such as saving preferences, temporary data, or any changes made by the user.
     */
    override fun onPause() {
        Log.i(TAG, "You are in the onPause() method.")
        closeCamera()
        stopBackgroundThread()
        super.onPause()
    }

    /**
    In Android, a TextureView is a view component
    that can be used to display content that is provided as a SurfaceTexture.
    This component is particularly useful when you want to display content from a camera preview,
    video playback, or other graphical content efficiently.
    The SurfaceTextureListener interface provides callbacks
    for events related to the SurfaceTexture associated with a TextureView.
    By implementing this interface, you can respond to events such as
    when the surface texture is available, updated, or destroyed.
     */
    private val surfaceListener = object : TextureView.SurfaceTextureListener{
        override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
            Log.d(TAG, "SurfaceTextureAvailable is called width: $p1 height: $p2")
            openCamera()
        }

        override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
            Log.d("SurfaceTexture", "Size has been changed")
        }

        override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean = true

        override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {

        }

    }

    /**
    Provides callbacks for various states and events related to a camera device
     */
    private val deviceStateCallback = object : CameraDevice.StateCallback(){
        override fun onOpened(camera : CameraDevice) {
            Log.d("onOpened", "camera device opened")
            cameraDevice = camera
            previewSession()
        }

        override fun onDisconnected(camera: CameraDevice) {
            Log.d("onDisconnected", "camera device disconnected")
            cameraDevice.close()
        }

        override fun onError(p0: CameraDevice, p1: Int) {
            Log.d("onError", "camera device error")
            cameraDevice.close()
            //finish()
        }

    }

    ///////////////////////////////////////////////////// ////////////////////////////////////////////////////////////////

    private fun openCamera(){
        checkCameraPermission()
    }

    //////////////////////////////////////////// Permissions //////////////////////////////////////////
    /**
    When this annotation is used, it tells EasyPermissions
    that after the specified permissions are granted,
    the annotated method (checkCameraPermission) should be executed automatically.
     */
    @AfterPermissionGranted(REQUEST_CAMERA_AND_STORAGE_PERMISSION)
    private fun checkCameraPermission() {
        val permissions = mutableListOf(Manifest.permission.CAMERA)

        if (needsStoragePermission()) {
            permissions.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }

        /*
         The * operator (spread operator) is used to spread the elements of an array
         when calling a function. In this context,
         it's used to pass each permission string from the array
         as separate arguments to the EasyPermissions.requestPermissions() function.
         */
        if (EasyPermissions.hasPermissions(activity, *permissions.toTypedArray())){
            Log.d("Permissions", "App has camera permission.")
            Log.d(TAG,"Version: ${Build.VERSION.SDK_INT}")
            connectCamera()
        } else{
            EasyPermissions.requestPermissions(activity!!, getString(R.string.camera_request), REQUEST_CAMERA_AND_STORAGE_PERMISSION, Manifest.permission.CAMERA)
            EasyPermissions.requestPermissions(activity!!, getString(R.string.camera_request), REQUEST_CAMERA_AND_STORAGE_PERMISSION, Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }
    }

    /**
    Check if WRITE_EXTERNAL_STORAGE permission is required based on conditions
     */
    private fun needsStoragePermission(): Boolean {
        // Add conditions if needed to determine if the permission is necessary.
        // For instance, you can check the device's API level or other factors.
        return Build.VERSION.SDK_INT <= Build.VERSION_CODES.P // Example condition
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    /**
    Connection to camera.
     */
    private fun connectCamera(){
        Log.e(TAG, "You are in connectCamera method.")
        val deviceId = cameraId(CameraCharacteristics.LENS_FACING_BACK)
        Log.d("Connect camera", "deviceId: $deviceId")
        try {
            cameraManager.openCamera(deviceId, deviceStateCallback, /*backgroundHandler*/ null)
            Log.i(TAG, "Camera were opened.")
        } catch (e: CameraAccessException){
            Log.e("Connect camera", e.toString())
        } catch (e: InterruptedException){
            Log.e("Connect camera", "Open camera device interrupted while opened")
        }
    }

    /**
    Returns the first id of the camera with the specific conditions(lens)
     */
    private fun cameraId (lens: Int) : String{
        var deviceId = listOf<String>()
        try {
            val cameraIdList = cameraManager.cameraIdList
            deviceId = cameraIdList.filter{
                lens == cameraCharacteristics(it, CameraCharacteristics.LENS_FACING)
            }
        } catch (e: CameraAccessException) {
            Log.d("CameraAccess", e.toString())
        }
        Log.d(TAG, "devices count is: ${deviceId.size}")
        // deviceId[0] == normal camera
        // deviceId[1] == 0.5 camera
        cameraId = deviceId[0]
        return cameraId
    }

    /**
    Checks if a camera have a specific key-characteristic
     */
    private fun <T> cameraCharacteristics(cameraId:String, key : CameraCharacteristics.Key<T>) : T? {
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)
        return when(key){
            CameraCharacteristics.LENS_FACING -> characteristics.get(key)
            CameraCharacteristics.SENSOR_ORIENTATION -> characteristics.get(key)
            CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES ->characteristics.get(key)
            CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP -> characteristics.get(key)
            CameraCharacteristics.SENSOR_INFO_MAX_FRAME_DURATION -> characteristics.get(key)
            CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE -> characteristics.get(key)
            CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE -> characteristics.get(key)


            else -> throw IllegalArgumentException("Key is not recognized")
        }
    }

    ///////////////////////////////////  Camera was connected   ////////////////////////////////////////

    private fun previewSession(){

        //val maxFrameDuration = cameraCharacteristics(cameraId, CameraCharacteristics.SENSOR_INFO_MAX_FRAME_DURATION)
        //val maxFps = 1e9 / maxFrameDuration!!.toDouble()
        //Log.i(TAG, "and here we go: $maxFps")
        // Calculation for optimal Surface size
        // TODO
        /**
         To calculate or determine the optimal Size for the preview.
         The mPreviewSize variable is assigned the calculated optimal size (if any).
         */
        val x = cameraCharacteristics(cameraId, CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val y = cameraCharacteristics(cameraId, CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        Log.i(TAG, "AGAGAGAGAGAGAGA: $x")

        val mPreviewSize : Size? = sizer()
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        val streamConfigMap = cameraCharacteristics(cameraId, CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)

        // Get supported output formats
        val outputFormats = streamConfigMap?.outputFormats

        // Get supported output sizes for MediaRecorder
        val outputSizes = streamConfigMap?.getOutputSizes(MediaRecorder::class.java)

        outputSizes?.forEach { size ->
            val minFrameDuration = streamConfigMap.getOutputMinFrameDuration(MediaRecorder::class.java, size)
            val frameRate = 1e9 / minFrameDuration.toFloat()

            Log.d(TAG, "Size: $size, Min Frame Duration: $minFrameDuration ns, Frame Rate: $frameRate fps")
        }

        val desiredSize = streamConfigMap?.getOutputSizes(MediaRecorder::class.java)?.firstOrNull { size ->
            size.width == 200 && size.height == 200
        }
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        val surfaceTexture = previewTextureView.surfaceTexture
        surfaceTexture?.setDefaultBufferSize(previewTextureView.width,previewTextureView.height)
        //surfaceTexture?.setDefaultBufferSize(mPreviewSize!!.width, mPreviewSize.height)
        //Log.d(TAG, "Preview Session is called!!! ${mPreviewSize!!.width} and ${mPreviewSize.height} and ${previewTextureView.height}")
        val surface = Surface(surfaceTexture)

        ///???????
        previewTextureView.layoutParams = ViewGroup.LayoutParams(previewTextureView.width, previewTextureView.height)

        // Choosing mode
        when(previewControlBool) {
            true -> {captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_MANUAL)}
            false -> {captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)}
        }

        captureRequestBuilder.addTarget(surface)

        cameraDevice.createCaptureSession(
            listOf(surface),
            object: CameraCaptureSession.StateCallback(){
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session

                   // desiredSize?.let { size ->
                        Log.i(TAG, "XXX ${isoValueProg}")
                        captureRequestBuilder.set(CaptureRequest.SENSOR_SENSITIVITY, isoValueProg) // ISO
                        captureRequestBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureTimeValueProg)
                        captureRequestBuilder.set(CaptureRequest.LENS_FOCUS_DISTANCE, focusValueProg)
                        captureRequestBuilder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(frameRateValueProg, frameRateValueProg))
                        //captureRequestBuilder.set(CaptureRequest.SCALER_CROP_REGION, Rect(0, 0, size.width, size.height)) // Set crop region
                    //}
                    setRepeatingRequest()
                }

                override fun onConfigureFailed(p0: CameraCaptureSession) {
                    Log.e("onConfigured", "creating capture session failed!")
                }
            }, backgroundHandler)
    }

    private fun seekBarExposureTimeProgressToProgValue(value: Int): Long{
        return (1_000_000_000/value).toLong()
    }


    /**
    backgroundThread (HandlerThread):
    backgroundThread is an instance of HandlerThread, which is a subclass of Thread designed
    to simplify background thread management. HandlerThread comes with an associated looper,
    which is used to create a Handler.
    The background thread created by backgroundThread will handle camera-related tasks.

    backgroundHandler (Handler):
    backgroundHandler is a Handler associated with the looper of the backgroundThread.
    You use this handler to post tasks to the background thread for execution.
    It allows you to perform camera operations or other time-consuming tasks without blocking the main thread.

    Message Queue:
    A message queue is a queue of tasks, also known as messages, that are scheduled
    to be executed on a specific thread. Each message contains a Runnable or a Handler along with additional data.

    Handler:
    A Handler is an Android class that allows you to enqueue tasks (messages) onto a thread's message queue.
    A Handler is associated with a specific Looper and thread, and it allows you to post tasks that will be executed on that thread.

    Looper:
    A Looper is responsible for managing the message queue of a thread.
    It continuously processes messages in the order they were added to the queue.
    When a Looper is active, it retrieves messages from the queue and dispatches them for execution.
     */
    private fun startBackgroundThread(){
        backgroundThread = HandlerThread("Camera2 Kotlin").also { it.start() }
        backgroundHandler = Handler(backgroundThread.looper)
    }


    /**
    The quitSafely() method is a safe way to request the background thread to quit. It posts a request for the background thread to quit its loop and finish its execution.
    It is a graceful way to stop the thread, allowing it to complete any pending tasks or clean up resources before exiting.

    The join() method is called on the backgroundThread. It blocks the calling thread
    (in this case, the current thread executing the function) until the backgroundThread has finished executing and has exited.
    This ensures that the background thread is fully stopped and cleaned up before the function continues executing.
     */
    private fun stopBackgroundThread(){
        backgroundThread.quitSafely()
        try {
            backgroundThread.join()
        } catch (e: InterruptedException) {
            Log.d("Quitting from thread", e.toString())
        }
    }

    /**
    There is information here about all possible sizes and frame rates for Video Capturing
     */
    private fun cameraInfo(){
        val streamConfigMap = cameraCharacteristics(cameraId, CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)

        // Retrieve the output sizes for video recording
        val outputSizes = streamConfigMap?.getOutputSizes(MediaRecorder::class.java)

        // Retrieve the supported frame rates
        outputSizes?.forEach { size ->
            val minFrameDuration =
                streamConfigMap.getOutputMinFrameDuration(MediaRecorder::class.java, size)
            val frameRate = 1e9 / minFrameDuration.toFloat()
            Log.d(TAG, "GGGGGGGGGGGGG: $frameRate  and Sizes: $size")
        }
    }

    private var imageReader: ImageReader? = null
    private val IMAGE_WIDTH = 1920 // Set the desired width

    private val IMAGE_HEIGHT = 1080 // Set the desired height

    private val MAX_IMAGES = 1
    private val capturedImages: MutableList<Image> = mutableListOf()
    private fun ShutterZeroLagSession(){

        // Calculation for optimal Surface size
        // TODO
        val surfaceTexture = previewTextureView.surfaceTexture
        surfaceTexture?.setDefaultBufferSize(previewTextureView.width, previewTextureView.height)
        val textureSurface = Surface(surfaceTexture)
        imageReader = ImageReader.newInstance(IMAGE_WIDTH, IMAGE_HEIGHT, ImageFormat.JPEG, MAX_IMAGES)
        imageReader?.setOnImageAvailableListener(
            { reader ->
                // Handle the availability of a new image
                val image = reader?.acquireLatestImage()
                // Process or save the image as needed
                image?.let { d -> capturedImages.add(d) }
                // Don't forget to close the image when done
                image?.close()
            },
            backgroundHandler
        )

        // Get the ImageReader surface
        val imageSurface: Surface = imageReader!!.surface

        when(previewControlBool) {
            true -> {captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_ZERO_SHUTTER_LAG)}
            false -> {captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_ZERO_SHUTTER_LAG)}
        }
        //captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
        //captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_MANUAL)
        captureRequestBuilder.addTarget(textureSurface)
        captureRequestBuilder.addTarget(imageSurface)
        val surfaces = java.util.ArrayList<Surface>().apply {
            add(textureSurface)
            add(imageSurface)
        }

        cameraDevice.createCaptureSession(
            surfaces,
            object: CameraCaptureSession.StateCallback(){
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    //captureRequestBuilder.set(CaptureRequest.SENSOR_SENSITIVITY, isoValueProg) // ISO
                    //captureRequestBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureTimeValueProg)
                    //captureRequestBuilder.set(CaptureRequest.LENS_FOCUS_DISTANCE, focusValueProg)
                    //captureRequestBuilder.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_INCANDESCENT) // WB
                    //captureRequestBuilder.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, 2) // EV
                    //captureRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                    /*captureRequestBuilder.set(
                        CaptureRequest.CONTROL_MODE,
                        CaptureRequest.CONTROL_MODE_AUTO
                    )*/
                    setRepeatingRequest()
                }

                override fun onConfigureFailed(p0: CameraCaptureSession) {
                    Log.e("onConfigured", "creating capture session failed!")
                }
            }, backgroundHandler)
    }

    private fun closeCamera(){
        if(::captureSession.isInitialized)
            captureSession.close()
        if(this::cameraDevice.isInitialized)
            cameraDevice.close()
    }

    private fun setupMediaRecorder() {
        /**
        get actual orientation of the display screen itself (not the sensor)
        Activity represents a single screen in an Android app, and it provides access to the application's UI and other resources.
        windowManager: windowManager is a property of the Activity that gives access to the window manager service.
        This service provides information about the display and window-related functions.
        defaultDisplay: defaultDisplay is a property of the window manager that represents the default display of the device.
        This display is typically the main screen of the device.
        rotation: Finally, rotation is a property of the default display that indicates the current rotation of the device's screen.
        It returns one of the constants Surface.ROTATION_0, Surface.ROTATION_90, Surface.ROTATION_180,
        or Surface.ROTATION_270, depending on how the device is currently oriented.
         */
        val rotation = activity?.windowManager?.defaultDisplay?.rotation
        val sensorOrientation = cameraCharacteristics(
            cameraId(CameraCharacteristics.LENS_FACING_BACK),
            CameraCharacteristics.SENSOR_ORIENTATION
        )
        Log.i(TAG, "HHHHHHHHHHHHHHH: $sensorOrientation")
        when (sensorOrientation){
            SENSOR_0_ORIENTATION_DEGREES ->
                mediaRecorder.setOrientationHint(ORIENTATION_0.get(rotation!!))
            SENSOR_90_ORIENTATION_DEGREES ->
                mediaRecorder.setOrientationHint(ORIENTATION_90.get(rotation!!))
            SENSOR_180_ORIENTATION_DEGREES ->
                mediaRecorder.setOrientationHint(ORIENTATION_180.get(rotation!!))
            SENSOR_270_ORIENTATION_DEGREES ->
                mediaRecorder.setOrientationHint(ORIENTATION_270.get(rotation!!))
        }
        mediaRecorder.apply {
            setVideoSource(MediaRecorder.VideoSource.SURFACE) /// CAMERA????
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setOutputFile(createVideoFile())
            setVideoEncodingBitRate(10000)
            // TODO framerate????
            setVideoFrameRate(30)
            //TODO???????????? why 1920 and 1080
            setVideoSize(1920, 1080)
            setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            prepare()
        }
    }

    private fun stopMediaRecorder(){
        mediaRecorder.apply {
            try {
                stop()
                reset()
            } catch (e: IllegalStateException) {
                Log.e("stopMediaRecorder", e.toString())
            }

        }
    }

    private fun setProgressBarToManualControl(){
        Log.d(TAG, "exposureTimeValueProg: $exposureTimeValueProg")
        isoSeekBar!!.progress = isoValueProg
        exposureTimeSeekBar!!.progress = seekBarExposureTimeProgressToProgValue(exposureTimeValueProg.toInt()).toInt()
        Log.d(TAG, "exposureTimeValueProg: ${exposureTimeSeekBar!!.progress}")
        focusSeekBar!!.progress = (focusValueProg * 100).toInt()
        frameRateSeekBar!!.progress = frameRateValueProg
        Log.d(TAG, "exposureTimeValueProg: $exposureTimeValueProg")
        isoSeekBarValueText!!.text = "ISO: $isoValueProg"
        exposureTimeSeekBarValueText!!.text = "ET: 1/${seekBarExposureTimeProgressToProgValue(exposureTimeValueProg.toInt())}"
        focusSeekBarValueText!!.text = "Focus: $focusValueProg"
        frameRateSeekBarValueText!!.text = "fps: $frameRateValueProg"
    }

    private fun setRepeatingRequest(){
        captureSession.setRepeatingRequest(captureRequestBuilder.build(), object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(
                session: CameraCaptureSession,
                request: CaptureRequest,
                result: TotalCaptureResult
            ) {
                super.onCaptureCompleted(session, request, result)

                // Setting prog values especially for listeners in onActivityCreated() method
                //Log.i(TAG, "AAAAAAAAAAAAAAAAAAAAAA ${result.get(CaptureResult.SCALER_CROP_REGION)}")
                exposureTimeValueProg = result.get(CaptureResult.SENSOR_EXPOSURE_TIME)!!
                isoValueProg = result.get(CaptureResult.SENSOR_SENSITIVITY)!!
                focusValueProg = result.get(CaptureResult.LENS_FOCUS_DISTANCE)!!
                frameRateValueProg =  (1/(result.get(CaptureResult.SENSOR_FRAME_DURATION)!! / 1e9)).toInt()

                // Adding current values on the screen
                requireActivity().runOnUiThread {
                    isoValueText!!.text = "ISO: $isoValueProg"
                    exposureTimeValueText!!.text = "Shutter speed: $exposureTimeValueProg"
                    focusValueText!!.text = "Focus: $focusValueProg"
                    frameRateValueText!!.text = "Frame Rate: $frameRateValueProg"
                }
            }
        }, backgroundHandler)
    }

    private fun startChronometer(){
        chronometer?.base = SystemClock.elapsedRealtime()
        chronometer?.setTextColor(resources.getColor(android.R.color.holo_red_light, null))
        chronometer?.start()
        /*timer.scheduleAtFixedRate(object : TimerTask() {
            override fun run() {
                handler.post {
                    Log.d(TAG, "")
                    exposureTimeValueProg = exposureTimeValueProg + 250_000
                    captureRequestBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureTimeValueProg)
                    setRepeatingRequest()
                }
            }
        }, 0, 1000) // Run every second*/
    }

    private fun stopChronometer(){
        chronometer!!.setTextColor(resources.getColor(android.R.color.white, null))
        chronometer!!.stop()
       // timer.cancel()
    }



    ////////////////////////////////TODO: Some strange methods////////////////////////////////////////////


    private fun sizer() : Size? {

        val characteristics : CameraCharacteristics = cameraManager.getCameraCharacteristics(cameraId)
        val map: StreamConfigurationMap? = characteristics.get(
            CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP
        )

        // Find out if we need to swap dimension to get the preview size relative to sensor
        // coordinate.
        val displayRotation = activity!!.windowManager.defaultDisplay.rotation
        //noinspection ConstantConditions
        mSensorOrientation = characteristics.get<Int>(CameraCharacteristics.SENSOR_ORIENTATION)!!
        var swappedDimensions = false
        when (displayRotation) {
            Surface.ROTATION_0, Surface.ROTATION_180 -> if (mSensorOrientation === 90 || mSensorOrientation === 270) {
                swappedDimensions = true
            }

            Surface.ROTATION_90, Surface.ROTATION_270 -> if (mSensorOrientation === 0 || mSensorOrientation === 180) {
                swappedDimensions = true
            }

            else -> Log.e(TAG, "Display rotation is invalid: $displayRotation")
        }
        Log.e(TAG, "swappedDimensions = $swappedDimensions")

        // It is an array of pairs (width, height) of all possible sizes for the camera
        val s = map!!.getOutputSizes(SurfaceTexture::class.java)
        for (i in 0..s.size-1){
            Log.e(TAG, "width = ${s.get(i).width} and height = ${s.get(i).height}")
        }

        val displaySize = Point()
        val windowManager = activity!!.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val display = windowManager.defaultDisplay
// Get the display size
        display.getSize(displaySize)
        val xCoordinate = displaySize.x
        val yCoordinate = displaySize.y
        Log.e(TAG, "display width = ${displaySize.x}   height = ${displaySize.y}")


        var rotatedPreviewWidth = width
        var rotatedPreviewHeight = height
        var maxPreviewWidth = displaySize.x
        var maxPreviewHeight = displaySize.y
        val largest = Collections.max(
            Arrays.asList(*map.getOutputSizes(SurfaceTexture::class.java)),
            CompareSizesByArea()
        )

        if (swappedDimensions) {
            rotatedPreviewWidth = height;
            rotatedPreviewHeight = width;
            maxPreviewWidth = displaySize.y;
            maxPreviewHeight = displaySize.x;
        }

                    if (maxPreviewWidth > MAX_PREVIEW_WIDTH) {
                        maxPreviewWidth = MAX_PREVIEW_WIDTH;
                    }

                    if (maxPreviewHeight > MAX_PREVIEW_HEIGHT) {
                        maxPreviewHeight = MAX_PREVIEW_HEIGHT;
                    }

        val mPreviewSize = chooseOptimalSize(
            map.getOutputSizes<SurfaceTexture>(SurfaceTexture::class.java),
            rotatedPreviewWidth, rotatedPreviewHeight, maxPreviewWidth,
            maxPreviewHeight, largest)

        val orientation = resources.configuration.orientation
        if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
            previewTextureView.layoutParams.height = mPreviewSize!!.height
            previewTextureView.layoutParams.width = mPreviewSize.width
        } else {
            previewTextureView.layoutParams.height = 360 //mPreviewSize!!.width
            previewTextureView.layoutParams.width = 720//mPreviewSize.height
        }

        Log.i(TAG, "Sizer is running@@@@@@@@@@@@@@@@@@@ ${previewTextureView.layoutParams.height}")
        return mPreviewSize
    }

    /**
     * Given `choices` of `Size`s supported by a camera, choose the smallest one that
     * is at least as large as the respective texture view size, and that is at most as large as the
     * respective max size, and whose aspect ratio matches with the specified value. If such size
     * doesn't exist, choose the largest one that is at most as large as the respective max size,
     * and whose aspect ratio matches with the specified value.
     *
     * @param choices           The list of sizes that the camera supports for the intended output
     * class
     * @param textureViewWidth  The width of the texture view relative to sensor coordinate
     * @param textureViewHeight The height of the texture view relative to sensor coordinate
     * @param maxWidth          The maximum width that can be chosen
     * @param maxHeight         The maximum height that can be chosen
     * @param aspectRatio       The aspect ratio
     * @return The optimal `Size`, or an arbitrary one if none were big enough
     */
    private fun chooseOptimalSize(
        choices: Array<Size>, textureViewWidth: Int,
        textureViewHeight: Int, maxWidth: Int, maxHeight: Int, aspectRatio: Size
    ): Size? {

        // Collect the supported resolutions that are at least as big as the preview Surface
        val bigEnough: MutableList<Size> = ArrayList()
        // Collect the supported resolutions that are smaller than the preview Surface
        val notBigEnough: MutableList<Size> = ArrayList()
        val w = aspectRatio.width
        val h = aspectRatio.height
        for (option in choices) {
            if (option.width <= maxWidth && option.height <= maxHeight && option.height == option.width * h / w) {
                if (option.width >= textureViewWidth &&
                    option.height >= textureViewHeight
                ) {
                    bigEnough.add(option)
                } else {
                    notBigEnough.add(option)
                }
            }
        }

        // Pick the smallest of those big enough. If there is no one big enough, pick the
        // largest of those not big enough.
        return if (bigEnough.size > 0) {
            Collections.min<Size>(bigEnough, CompareSizesByArea())
        } else if (notBigEnough.size > 0) {
            Collections.max<Size>(notBigEnough, CompareSizesByArea())
        } else {
            Log.e(TAG, "Couldn't find any suitable preview size")
            choices[0]
        }
    }


    /**
     * Compares two `Size`s based on their areas.
     */
    internal class CompareSizesByArea : Comparator<Size?> {
        override fun compare(lhs: Size?, rhs: Size?): Int {
            // We cast here to ensure the multiplications won't overflow
            return java.lang.Long.signum(
                lhs!!.width.toLong() * lhs.height -
                        rhs!!.width.toLong() * rhs.height
            )
        }

    }

    ////////////////////////////////////////////////////////////////////////////

    private fun createVideoFileName() : String{
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        return "VIDEO_${timeStamp}.mp4"
    }

    private fun createVideoFile() : File{
        // TODO directory
        val videoDirectory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM)
        if (!videoDirectory.exists()) {
            videoDirectory.mkdirs()
        }
        val videoFile = File(videoDirectory, createVideoFileName())
        currentVideoFilePath = videoFile.absolutePath
        Log.d(TAG, "File was created")
        return videoFile
    }

    private fun triggerMediaScan(videoUri: Uri) {
        MediaScannerConnection.scanFile(context, arrayOf(videoUri.path), null) { _, _ ->
            // Media scan completed
        }
    }

    private fun recordSession(){

        Log.d("RecordSession", "Record Session is called!!!")
        /*
        if(!isRecording)
            setupMediaRecorder()
*/
        val surfaceTexture = previewTextureView.surfaceTexture
        surfaceTexture?.setDefaultBufferSize(previewTextureView.width, previewTextureView.height)
        val textureSurface = Surface(surfaceTexture)
        val recordSurface = mediaRecorder.surface

        when(previewControlBool) {
            true -> {captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_MANUAL)
                Log.d(TAG, "we are here")}
            false -> {captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
                Log.d(TAG, "we are here")}
        }
        captureRequestBuilder.addTarget(textureSurface)
        captureRequestBuilder.addTarget(recordSurface)
        val surfaces = java.util.ArrayList<Surface>().apply {
            add(textureSurface)
            add(recordSurface)
        }


        cameraDevice.createCaptureSession(
            surfaces,
            object: CameraCaptureSession.StateCallback(){
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    captureRequestBuilder.set(CaptureRequest.SENSOR_SENSITIVITY, isoValueProg) // ISO
                    captureRequestBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureTimeValueProg) // Shutter speed
                    captureRequestBuilder.set(CaptureRequest.LENS_FOCUS_DISTANCE, focusValueProg)
                    captureRequestBuilder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(frameRateValueProg, frameRateValueProg))
                    //captureRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)
                    //captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)

                    //captureRequestBuilder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(90, 90))

                    /*captureRequestBuilder.set(
                        CaptureRequest.CONTROL_MODE,
                        CaptureRequest.CONTROL_MODE_OFF
                    )*/
                    //captureRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                    setRepeatingRequest()
                    //captureSession.setRepeatingRequest(captureRequestBuilder.build(), null, null /*backgroundHandler*/)
                    /*val captureRequest = captureRequestBuilder.build()
                    captureSession.setRepeatingRequest(captureRequest, null, null)*/
                    if(!isRecording)
                        mediaRecorder.start()
                    isRecording = true
                }

                override fun onConfigureFailed(p0: CameraCaptureSession) {
                    Log.e("onConfigured", "creating record session failed!")

                }

            },backgroundHandler)
    }

    private fun startRecordSession(){
        startChronometer()
        recordSession()
    }

    private fun stopRecordSession(){
        stopChronometer()
        stopMediaRecorder()
        previewSession()
        thumbButton?.setImageDrawable(createRoundThumb())
        cameraViewModel.videoUri = Uri.fromFile(File(currentVideoFilePath))
        triggerMediaScan(Uri.fromFile(File(currentVideoFilePath)))
    }

    ////////////////////////////////////////////////////////////////////////////
    // Thumbnail

    private fun createVideoThumb() = ThumbnailUtils.createVideoThumbnail(currentVideoFilePath, MediaStore.Video.Thumbnails.MICRO_KIND)

    // Convert to around thumbnail
    private fun createRoundThumb() : RoundedBitmapDrawable{
        var drawable = RoundedBitmapDrawableFactory.create(resources, createVideoThumb())
        drawable.isCircular = true
        return drawable
    }

    private fun replaceFragment(fragment: Fragment){
        val fragmentTransaction = activity?.supportFragmentManager?.beginTransaction()
        fragmentTransaction?.replace(R.id.fragmentContainer, fragment)
        fragmentTransaction?.addToBackStack(null)
        fragmentTransaction?.commit()
    }

}
