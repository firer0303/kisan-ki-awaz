# Kisan Ki Awaz - ProGuard Rules
-keepattributes *Annotation*
-keep class com.kisanKiAwaz.app.** { *; }
-keepclassmembers class * { @android.webkit.JavascriptInterface <methods>; }
-dontwarn okhttp3.**
-dontwarn okio.**
