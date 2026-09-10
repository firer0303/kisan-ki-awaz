package com.kisanKiAwaz.app

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import android.util.Base64
import android.util.Log
import android.webkit.*
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.util.*

/**
 * Kisan Ki Awaz - Main Activity
 *
 * Hosts a WebView that connects to the FastAPI backend.
 * Provides native Android bridge for camera, microphone, and TTS.
 */
class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    companion object {
        private const val TAG = "KisanKiAwaz"
        private const val REQUEST_CAMERA = 1001
        private const val REQUEST_AUDIO = 1002
    }

    private lateinit var webView: WebView
    private var tts: TextToSpeech? = null

    // API base URL - injected from BuildConfig (Gradle buildConfigField)
    private val apiBaseUrl: String by lazy { BuildConfig.API_BASE_URL }

    // Activity result launchers
    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { handleImageUri(it) }
    }

    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap: Bitmap? ->
        bitmap?.let { handleCameraBitmap(it) }
    }

    private val speechLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val matches = result.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            val text = matches?.firstOrNull() ?: ""
            if (text.isNotEmpty()) {
                // Pass result to web app and auto-submit
                val js = "window.onVoiceResult && window.onVoiceResult('${escapeJsString(text)}');"
                webView.evaluateJavascript(js, null)
            } else {
                Toast.makeText(this, "No speech recognized", Toast.LENGTH_SHORT).show()
            }
        } else {
            // Notify web app that listening stopped
            webView.evaluateJavascript("document.getElementById('voiceBtn')?.classList.remove('listening');", null)
        }
    }

    // ──────────────────────────────────────────────────────────
    // Lifecycle
    // ──────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        webView = findViewById(R.id.webView)
        setupWebView()
        setupTTS()
        setupBackNavigation()

        // Load the web frontend from the API server
        webView.loadUrl(apiBaseUrl)
    }

    override fun onDestroy() {
        tts?.stop()
        tts?.shutdown()
        super.onDestroy()
    }

    private fun setupBackNavigation() {
        val callback = object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (webView.canGoBack()) {
                    webView.goBack()
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                    isEnabled = true
                }
            }
        }
        onBackPressedDispatcher.addCallback(this, callback)
    }

    // ──────────────────────────────────────────────────────────
    // WebView Setup
    // ──────────────────────────────────────────────────────────
    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            mediaPlaybackRequiresUserGesture = false  // Allow auto-narration
            allowFileAccess = true
            allowContentAccess = true
            useWideViewPort = true
            loadWithOverviewMode = true
            setSupportZoom(false)
            builtInZoomControls = false
            displayZoomControls = false
            cacheMode = WebSettings.LOAD_DEFAULT
            mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
        }

        // Enable debugging in debug builds
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG)

        // JavaScript bridge for native features
        webView.addJavascriptInterface(AndroidBridge(), "AndroidBridge")

        // Chrome client for file inputs / permissions
        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest?) {
                request?.grant(request.resources)
            }
        }

        // Handle errors with user-friendly fallback
        webView.webViewClient = object : WebViewClient() {
            override fun onReceivedError(
                view: WebView?, request: WebResourceRequest?,
                error: WebResourceError?
            ) {
                Log.e(TAG, "WebView error: ${error?.description}")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    showErrorInWebView("Connection error: ${error?.description}. Please check internet and try again.")
                }
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                Log.i(TAG, "Page loaded: $url")
            }
        }
    }

    private fun showErrorInWebView(message: String) {
        val js = """
            (function(){
                var box = document.createElement('div');
                box.style.cssText = 'background:#FFEBEE;border-left:4px solid #F44336;padding:16px;margin:16px;border-radius:8px;color:#B71C1C;font-family:sans-serif;';
                box.innerHTML = '<strong>⚠️ ' + ${escapeJsString(message).let { "'$it'" }} + '</strong><br><button onclick="location.reload()" style="margin-top:10px;padding:8px 16px;border:none;background:#1B5E20;color:#fff;border-radius:6px;">Retry</button>';
                document.body.insertBefore(box, document.body.firstChild);
            })();
        """.trimIndent()
        webView.evaluateJavascript(js, null)
    }

    // ──────────────────────────────────────────────────────────
    // Text-to-Speech
    // ──────────────────────────────────────────────────────────
    private fun setupTTS() {
        tts = TextToSpeech(this, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.forLanguageTag("en-US"))
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.w(TAG, "TTS language not supported, using default")
            }
        }
    }

    private fun speakText(text: String, langCode: String) {
        val locale = when (langCode) {
            "ur" -> Locale("ur", "PK")
            "sd" -> Locale("ur", "PK")  // Fallback: Sindhi -> Urdu
            "pa" -> Locale("ur", "PK")  // Fallback: Punjabi -> Urdu
            "ps" -> Locale("ur", "PK")  // Fallback: Pashto -> Urdu
            "bal" -> Locale("ur", "PK") // Fallback: Balochi -> Urdu
            else -> Locale.ENGLISH
        }
        tts?.setLanguage(locale)
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "narration_${System.currentTimeMillis()}")
    }

    // ──────────────────────────────────────────────────────────
    // Image Handling
    // ──────────────────────────────────────────────────────────
    private fun handleImageUri(uri: Uri) {
        try {
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = android.graphics.BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            bitmap?.let { handleCameraBitmap(it) }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load image: ${e.message}")
            Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
        }
    }

    private fun handleCameraBitmap(bitmap: Bitmap) {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 85, stream)
        val base64 = Base64.encodeToString(stream.toByteArray(), Base64.NO_WRAP)

        // Pass image to WebView using the global setter
        val js = """
            (function() {
                if (window.setCapturedImage) {
                    window.setCapturedImage('$base64');
                } else {
                    // Fallback for older frontend
                    window.selectedImageBase64 = '$base64';
                    var preview = document.getElementById('imagePreview');
                    if (preview) {
                        preview.src = 'data:image/jpeg;base64,$base64';
                        preview.classList.add('visible');
                    }
                    var btn = document.getElementById('analyzeBtn');
                    if (btn) btn.classList.remove('hidden');
                }
            })();
        """.trimIndent()
        webView.evaluateJavascript(js, null)
    }

    // ──────────────────────────────────────────────────────────
    // Permissions
    // ──────────────────────────────────────────────────────────
    private fun checkAndRequestPermission(permission: String, requestCode: Int): Boolean {
        if (ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED) {
            return true
        }
        ActivityCompat.requestPermissions(this, arrayOf(permission), requestCode)
        return false
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            REQUEST_CAMERA -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    cameraLauncher.launch(null)
                } else {
                    Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
                }
            }
            REQUEST_AUDIO -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "Permission granted. Tap voice button again.", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "Microphone permission is required for voice input", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────────
    // JavaScript Bridge - exposes native Android functions to WebView
    // ──────────────────────────────────────────────────────────
    inner class AndroidBridge {

        @JavascriptInterface
        fun showToast(message: String) {
            runOnUiThread {
                Toast.makeText(this@MainActivity, message, Toast.LENGTH_SHORT).show()
            }
        }

        @JavascriptInterface
        fun openCamera() {
            if (checkAndRequestPermission(Manifest.permission.CAMERA, REQUEST_CAMERA)) {
                runOnUiThread { cameraLauncher.launch(null) }
            }
        }

        @JavascriptInterface
        fun openGallery() {
            runOnUiThread { imagePickerLauncher.launch("image/*") }
        }

        @JavascriptInterface
        fun startSpeechRecognition(langCode: String) {
            if (checkAndRequestPermission(Manifest.permission.RECORD_AUDIO, REQUEST_AUDIO)) {
                val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE, getSpeechLocale(langCode))
                    putExtra(RecognizerIntent.EXTRA_PROMPT, "Speak your farming question...")
                }
                runOnUiThread { speechLauncher.launch(intent) }
            }
        }

        @JavascriptInterface
        fun speak(text: String, langCode: String) {
            runOnUiThread { speakText(text, langCode) }
        }

        @JavascriptInterface
        fun stopSpeaking() {
            tts?.stop()
        }

        @JavascriptInterface
        fun getApiBaseUrl(): String = apiBaseUrl

        @JavascriptInterface
        fun isRunningInApp(): Boolean = true
    }

    private fun getSpeechLocale(langCode: String): String {
        return when (langCode) {
            "ur" -> "ur-PK"
            "sd" -> "ur-PK"   // Android STT has limited Sindhi; fallback to Urdu
            "pa" -> "pa-IN"
            "ps" -> "ps-AF"
            "bal" -> "ur-PK"  // Fallback to Urdu for Balochi
            else -> "en-US"
        }
    }

    private fun escapeJsString(input: String): String {
        return input
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
    }
}
