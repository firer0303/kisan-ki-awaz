# Kisan Ki Awaz - Public Deployment Guide

To make the Android APK usable by anyone, the FastAPI backend must be hosted on a public URL.

## Option 1: Render.com (Recommended for beginners)

1. Push this project to GitHub.
2. Go to [render.com](https://render.com) and create a free account.
3. Click **New +** → **Web Service** → connect your GitHub repo.
4. Render will detect `render.yaml` and use Docker.
5. After deployment, copy your public URL (e.g., `https://kisan-ki-awaz-api.onrender.com`).
6. Replace `API_BASE_URL` in `android/app/build.gradle` with that URL.
7. Rebuild the APK: `gradlew assembleDebug`

> Note: Free Render plans spin down after inactivity. First request may take 30-60 seconds.

## Option 2: Fly.io

1. Install [flyctl](https://fly.io/docs/hands-on/install-flyctl/).
2. Run `fly launch` in this folder (it will use `fly.toml`).
3. Run `fly deploy`.
4. Copy the public URL and update `android/app/build.gradle`.
5. Rebuild the APK.

## Option 3: Alibaba Cloud (for production scale)

1. Create an ECS instance or Function Compute service in Alibaba Cloud console.
2. Upload the Docker image built from `Dockerfile`.
3. Expose port 8000 with a public IP / domain.
4. Set environment variables for API keys.
5. Update `API_BASE_URL` in `android/app/build.gradle` and rebuild.

## Option 4: ngrok (instant temporary public URL)

For quick demos without deploying:

1. Sign up at [ngrok.com](https://ngrok.com) and get an auth token.
2. Install ngrok and run:
   ```bash
   ngrok config add-authtoken YOUR_TOKEN
   ngrok http 8000
   ```
3. Copy the `https://xxxx.ngrok-free.app` URL.
4. Update `API_BASE_URL` in `android/app/build.gradle`.
5. Rebuild the APK.

## Updating the APK URL

Edit `kisan_ki_awaz/android/app/build.gradle`:

```gradle
buildConfigField "String", "API_BASE_URL", "\"https://your-public-url.com\""
```

Then rebuild:

```powershell
cd kisan_ki_awaz/android
gradlew assembleDebug
```

The APK will be at `app/build/outputs/apk/debug/app-debug.apk`.

## Required Environment Variables

Set these in your cloud dashboard or `.env` file:

- `OPENAI_API_KEY` — for LLM responses
- `WEATHER_API_KEY` — for weather data (optional, demo data used if missing)

## Important Notes

- The backend uses HTTP (not HTTPS) for local development. Public deployments should use HTTPS.
- The `network_security_config.xml` currently allows cleartext traffic. For production, remove `android:usesCleartextTraffic="true"` and configure SSL.
- The APK built with a debug key cannot be uploaded to Google Play. For Play Store release, create a release build with a signing key.
