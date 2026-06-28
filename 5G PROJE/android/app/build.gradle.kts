// app/build.gradle.kts — Sinaptic5G Android Uygulama Yapılandırması
// compileSdk 36 — android-36.1 ile eşleştirildi (11.06.2026)

import groovy.json.JsonSlurper
import java.security.MessageDigest

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.devtools.ksp")           // Room annotation processing
}

android {
    namespace   = "com.sinaptic.tripwire"
    compileSdk  = 36

    defaultConfig {
        applicationId = "com.sinaptic.tripwire"
        minSdk        = 26      // Android 8.0 — CameraX + WebSocket desteği
        targetSdk     = 36
        versionCode   = 1
        versionName   = "1.0.0"
        manifestPlaceholders["appAuthRedirectScheme"] = "com.sinaptic.tripwire"
        manifestPlaceholders["usesCleartextTraffic"] = "false"

        buildConfigField("String", "API_BASE_URL",
            "\"${project.findProperty("API_BASE_URL") ?: ""}\"")
        buildConfigField("String", "MEDIA_BASE_URL",
            "\"${project.findProperty("MEDIA_BASE_URL") ?: ""}\"")
        buildConfigField("String", "OIDC_AUTHORIZATION_ENDPOINT",
            "\"${project.findProperty("OIDC_AUTHORIZATION_ENDPOINT") ?: ""}\"")
        buildConfigField("String", "OIDC_TOKEN_ENDPOINT",
            "\"${project.findProperty("OIDC_TOKEN_ENDPOINT") ?: ""}\"")
        buildConfigField("String", "OIDC_CLIENT_ID",
            "\"${project.findProperty("OIDC_CLIENT_ID") ?: ""}\"")
        buildConfigField("String", "OIDC_REDIRECT_URI",
            "\"${project.findProperty("OIDC_REDIRECT_URI") ?: "com.sinaptic.tripwire:/oauth2redirect"}\"")
    }

    buildTypes {
        debug {
            isDebuggable    = true
            isMinifyEnabled = false
            manifestPlaceholders["usesCleartextTraffic"] = "false"
        }
        release {
            isMinifyEnabled   = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            manifestPlaceholders["usesCleartextTraffic"] = "false"
        }
    }

    buildFeatures {
        viewBinding    = true
        buildConfig    = true
        mlModelBinding = true
    }

    androidResources {
        noCompress += "tflite"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

val verifyModelAsset by tasks.registering {
    group = "verification"
    description = "Verifies the TFLite artifact hash and locked tensor contract."
    doLast {
        val assetsDir = file("src/main/assets")
        val manifestFile = assetsDir.resolve("model_manifest.json")
        val lockFile = rootProject.file("model_lock.json")
        check(manifestFile.isFile) { "model_manifest.json is missing" }
        check(lockFile.isFile) { "model_lock.json is missing" }
        val manifest = JsonSlurper().parse(manifestFile) as Map<*, *>
        val lock = JsonSlurper().parse(lockFile) as Map<*, *>
        val asset = assetsDir.resolve(manifest["model_asset"].toString())
        check(asset.isFile) { "TFLite model asset is missing: ${asset.name}" }
        val digest = MessageDigest.getInstance("SHA-256")
        asset.inputStream().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        val actualHash = digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
        check(actualHash == manifest["artifact_sha256"] && actualHash == lock["artifact_sha256"]) {
            "TFLite SHA-256 does not match manifest/model lock"
        }
        check(asset.length() == (manifest["artifact_size_bytes"] as Number).toLong()) {
            "TFLite size does not match manifest"
        }
        listOf("input_shape", "input_dtype", "output_shape", "output_dtype").forEach { key ->
            check(manifest[key].toString() == lock[key].toString()) {
                "TFLite tensor contract mismatch for $key"
            }
        }
        check(assetsDir.listFiles { file -> file.extension == "tflite" }?.size == 1) {
            "APK must contain exactly one TFLite asset"
        }
        check(assetsDir.walkTopDown().none { it.extension.equals("onnx", ignoreCase = true) }) {
            "APK assets must not contain ONNX"
        }
    }
}

tasks.named("preBuild").configure { dependsOn(verifyModelAsset) }

dependencies {
    val cameraxVersion = "1.4.1"
    val roomVersion    = "2.6.1"

    // ── CameraX ──────────────────────────────────────────────────────────
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")
    implementation("androidx.camera:camera-video:$cameraxVersion")

    // ── Tek mobil çıkarım backend'i: LiteRT/TFLite + NNAPI ───────────────
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("net.openid:appauth:0.11.1")

    // ── Room ─────────────────────────────────────────────────────────────
    implementation("androidx.room:room-runtime:$roomVersion")
    implementation("androidx.room:room-ktx:$roomVersion")
    ksp("androidx.room:room-compiler:$roomVersion")

    // ── Networking ───────────────────────────────────────────────────────
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("io.github.webrtc-sdk:android:137.7151.05")

    // ── Kotlin ───────────────────────────────────────────────────────────
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // ── AndroidX ─────────────────────────────────────────────────────────
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.6")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // ── Test ─────────────────────────────────────────────────────────────
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}
