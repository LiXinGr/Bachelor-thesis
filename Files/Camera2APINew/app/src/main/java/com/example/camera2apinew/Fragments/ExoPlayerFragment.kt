package com.example.camera2apinew.Fragments

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.common.util.Util
import androidx.media3.datasource.DefaultDataSource
import androidx.media3.datasource.DefaultDataSourceFactory
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.SimpleExoPlayer
import androidx.media3.exoplayer.source.ProgressiveMediaSource
import androidx.media3.exoplayer.trackselection.AdaptiveTrackSelection
import androidx.media3.exoplayer.trackselection.DefaultTrackSelector
import androidx.media3.exoplayer.upstream.DefaultBandwidthMeter
import androidx.media3.extractor.Extractor
import androidx.media3.extractor.ExtractorsFactory
import androidx.media3.ui.PlayerView
import androidx.navigation.fragment.findNavController
import com.example.camera2apinew.CameraViewModel
import com.example.camera2apinew.R

class ExoPlayerFragment : Fragment(){



    private val cameraViewModel by lazy {
        ViewModelProvider(requireActivity()).get(CameraViewModel::class.java)
    }

    /*override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_exoplayer, container, false)
    }*/

    companion object{
        val TAG = ExoPlayerFragment::class.java.name
        @JvmStatic fun newInstance() = ExoPlayerFragment()
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        Log.d(TAG, "Video uri: ${cameraViewModel.videoUri}")
    }

    private var playerView : PlayerView? = null
    @androidx.annotation.OptIn(androidx.media3.common.util.UnstableApi::class)
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val rootView = inflater.inflate(R.layout.fragment_exoplayer, container, false)
        playerView = rootView.findViewById(R.id.exoplayerView)

        val player = ExoPlayer.Builder(context!!).build()
        playerView?.player = player

        // Build the media item.
        val mediaItem = MediaItem.fromUri(cameraViewModel.videoUri!!)
        // Set the media item to be played.
        player.setMediaItem(mediaItem)
        // Prepare the player.
        player.prepare()
        // Start the playback.
        player.play()
        //Log.d(TAG, "Video uri: ${cameraViewModel.videoUri}")

        return rootView
    }

    /*private lateinit var playerView: PlayerView

    private val cameraViewModel by lazy {
        ViewModelProvider(requireActivity()).get(CameraViewModel::class.java)
    }

/*
    private val bandwidthMeter by lazy {
        DefaultBandwidthMeter.Builder(context!!)
    }

    private val trackSelectionFactory by lazy {
        AdaptiveTrackSelection.Factory(bandwidthMeter)
    }

    private val trackSelection by lazy {
        DefaultTrackSelector(trackSelectionFactory)
    }

    private val simpleExoPlayer by lazy {
        ExoPlayerFactory.newSimpleInstance(context, trackSelection)
    }

    private val applicationName by lazy {
        context?.applicationInfo?.loadLabel(context?.packageManager).toString()
    }

    private val dataSourceFactory by lazy {
        DefaultDataSourceFactory(context, Util.getUserAgent(context, applicationName))
    }

    private val videoMediaSource by lazy {
        ExtractorMediaSource.Factory(dataSourceFactory)
            .createMediaSource(cameraViewModel.videoUri)
    }

    private fun startExoPlayer(){
        exoplayerView.con
    }


*/
@androidx.annotation.OptIn(androidx.media3.common.util.UnstableApi::class)
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val rootView = inflater.inflate(R.layout.fragment_exoplayer, container, false)
        playerView = rootView.findViewById(R.id.exoplayerView)

        val player = ExoPlayer.Builder(context!!).build()
        playerView.player = null//player

        // Build the media item.
        val mediaItem = MediaItem.fromUri(cameraViewModel.videoUri!!)
        // Set the media item to be played.
        player.setMediaItem(mediaItem)
        // Prepare the player.
        player.prepare()
        // Start the playback.
        player.play()
    //Log.d(TAG, "Video uri: ${cameraViewModel.videoUri}")

        return rootView
    }

    override fun onActivityCreated(savedInstanceState: Bundle?) {
        super.onActivityCreated(savedInstanceState)

    }

    companion object {
        val TAG = ExoPlayerFragment::class.qualifiedName
        @JvmStatic fun newInstance() = ExoPlayerFragment()
    }




    private fun replaceFragment(fragment: Fragment){
        val fragmentTransaction = activity?.supportFragmentManager?.beginTransaction()
        fragmentTransaction?.replace(R.id.fragmentContainer, fragment)
        fragmentTransaction?.addToBackStack(null)
        fragmentTransaction?.commit()
    }*/

}