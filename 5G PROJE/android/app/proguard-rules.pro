# TFLite modelleri koru — obfuscation'dan muaf tut
-keep class org.tensorflow.** { *; }
-keep class com.google.android.** { *; }
-dontwarn org.tensorflow.**

# OkHttp / WebSocket
-dontwarn okhttp3.**
-keep class okhttp3.** { *; }
-keep interface okhttp3.** { *; }

# Room
-keep class * extends androidx.room.RoomDatabase
-keep @androidx.room.Entity class *
-keepclassmembers @androidx.room.Dao interface * { *; }

# Kotlin coroutines
-keepclassmembernames class kotlinx.** { volatile <fields>; }

# Data sınıfları koru
-keep class com.sinaptic.tripwire.model.** { *; }
