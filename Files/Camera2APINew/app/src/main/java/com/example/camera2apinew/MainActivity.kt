package com.example.camera2apinew

import android.content.res.Configuration
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.Menu
import android.view.MenuItem
import androidx.appcompat.app.ActionBarDrawerToggle
import androidx.appcompat.widget.Toolbar
import androidx.core.view.GravityCompat
import androidx.drawerlayout.widget.DrawerLayout
import androidx.fragment.app.Fragment
import androidx.viewpager.widget.ViewPager
import com.example.camera2apinew.Fragments.PreviewFragment
import com.example.camera2apinew.R
import com.google.android.material.navigation.NavigationView
import java.lang.ClassCastException

class MainActivity : AppCompatActivity() {

    var drawerLayout : DrawerLayout? = null
    var navigationView : NavigationView? = null
    var viewPager : ViewPager? = null
    var toolbar : Toolbar? = null

    // The ActionBarDrawerToggle is a utility class that connects
    // a navigation drawer with the action bar (or toolbar) in your app.
    val drawerToogle by lazy{
        ActionBarDrawerToggle(this, drawerLayout, toolbar , R.string.drawer_open, R.string.drawer_close)
    }

    override fun onBackPressed() {
        replaceFragment(PreviewFragment())
        //super.onBackPressed()
    }



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        drawerLayout = findViewById(R.id.drawerLayout)
        navigationView = findViewById(R.id.navigationView)
        toolbar = findViewById(R.id.toolbar)
        viewPager = findViewById(R.id.viewPager)


        // Set Up ActionBar:
        // In your activity's onCreate() method, set the Toolbar as the ActionBar.
        setSupportActionBar(toolbar)

        drawerLayout?.addDrawerListener(drawerToogle)

        navigationView?.setNavigationItemSelectedListener {
            selectDrawerItem(it)
            true
        }

        //val pagerAdapter = ImageFragmentPagerAdapter(supportFragmentManager)
        //viewPager?.adapter = pagerAdapter

         val fragment = PreviewFragment.newInstance()
        replaceFragment(fragment)
    }

    /*
    The onPostCreate method is a callback method in the Android Activity lifecycle.
    It's called after onCreate has completed execution and is often used in conjunction
    with ActionBarDrawerToggle to complete the setup of the navigation drawer's toggle
    icon after the activity's UI has been initialized. This method is commonly overridden
    in activities that use a navigation drawer to ensure proper synchronization between
    the navigation drawer's state and the action bar (or toolbar) icon.
     */
    override fun onPostCreate(savedInstanceState: Bundle?) {
        super.onPostCreate(savedInstanceState)
        // This line syncs the state of the ActionBarDrawerToggle with the state of the navigation drawer.
        // It ensures that the navigation icon's appearance (hamburger or back arrow)
        // matches the current state of the drawer (open or closed).
        drawerToogle.syncState()
    }

    /*
    In Android, the onConfigurationChanged method is a callback that is called
    when a change in the device's configuration occurs,
    such as a change in screen orientation, language, keyboard availability, and other configuration parameters.
    This callback allows you to respond to configuration changes and update your app's UI accordingly.
     */
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        drawerToogle.onConfigurationChanged(newConfig)
    }

    private fun selectDrawerItem(item: MenuItem){
        var fragment: Fragment?  = null
        /*
        The reason you need to use ::class.java here is that the when expression expects
        to work with the Java Class object, and the ::class.java syntax
        allows you to obtain that Java class representation from the Kotlin class.
         */
        // TODO: the first option
        /*val fragmentClass = when (item.itemId){
            R.id.firstFragmentItem -> FirstImageProject::class.java
            R.id.secondFragmentItem -> SecondImageProject::class.java
            else -> FirstImageProject::class.java
        }

        try {
            fragment = fragmentClass.newInstance()
        } catch (e: ClassCastException) {
            e.printStackTrace()
        }
        replaceFragment(fragment)*/
        // TODO: the second option
        /*when(item.itemId){
            R.id.firstFragmentItem -> viewPager?.currentItem = 0
            R.id.secondFragmentItem -> viewPager?.currentItem = 1
            else -> viewPager?.currentItem = 0
        }*/
        drawerLayout?.closeDrawer(GravityCompat.START)
    }


    // The onOptionsItemSelected(MenuItem item) method is a callback
    // that is called when an item from the options menu is selected
    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        /*return when(item.itemId){
            R.id.firstFragmentItem -> {
                val fragment = FirstImageProject.newInstance()
                replaceFragment(fragment)
                true
            }
            R.id.secondFragmentItem -> {
                val fragment = SecondImageProject.newInstance()
                replaceFragment(fragment)
                true
            }
            else -> super.onOptionsItemSelected(item)
        }*/
        return if(drawerToogle.onOptionsItemSelected(item)) true else super.onOptionsItemSelected(item)
    }

    // onCreateOptionsMenu is a method in the Android Activity
    // or Fragment lifecycle that you can override to create and populate the options menu for your app.
    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.fragment_menu, menu)
        return true
    }

    private fun replaceFragment(fragment: Fragment?){
        // method returns an instance of the FragmentTransaction class
        val fragmentTransaction = supportFragmentManager.beginTransaction()
        fragmentTransaction.replace(R.id.fragmentContainer, fragment!!)
        fragmentTransaction.commit()
    }
}