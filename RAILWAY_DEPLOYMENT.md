# Adcast Backend - Railway Deployment Guide

This guide will help you deploy your Adcast backend to Railway.app for remote testing.

## Prerequisites

- Git repository initialized (Railway deploys from GitHub)
- Railway account (free tier available)
- API keys for Anthropic and ElevenLabs

## Step 1: Push Code to GitHub

If you haven't already, push your code to GitHub:

```bash
cd /Users/tomhall-taylor
git init
git add adcast-mvp-backend.py requirements.txt railway.json .env.example
git commit -m "Prepare Adcast backend for Railway deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/adcast-backend.git
git push -u origin main
```

## Step 2: Set Up Railway

1. Go to [railway.app](https://railway.app)
2. Sign up or log in with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your `adcast-backend` repository
6. Railway will auto-detect Python and use the `railway.json` config

## Step 3: Configure Environment Variables

In the Railway dashboard:

1. Go to your project
2. Click on the "Variables" tab
3. Add these environment variables:

```
ANTHROPIC_API_KEY=your_anthropic_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
YOUTUBE_API_KEY=your_youtube_key_here (optional)
JWT_SECRET_KEY=your-production-secret-key-here
```

**Note:** Railway automatically provides `PORT` - don't set it manually.

## Step 4: Deploy

Railway will automatically deploy your app. You'll see:
- Build logs
- Deploy logs  
- Your app URL (e.g., `https://adcast-backend-production.up.railway.app`)

## Step 5: Update Mobile App Configuration

Once deployed, update the mobile app to use Railway:

1. Open `/Users/tomhall-taylor/adcast-mobile/src/config/api.config.ts`
2. Update `RAILWAY_URL` with your actual Railway URL:
   ```typescript
   const RAILWAY_URL = 'https://your-app-name.up.railway.app';
   ```
3. Set `USE_PRODUCTION = true` to switch to Railway
4. Restart Expo: `npx expo start`

## Step 6: Test the Deployment

1. Open your mobile app
2. Try creating a podcast
3. Check Railway logs for any errors

## Switching Between Local and Production

Edit `src/config/api.config.ts`:

- **Local development:** `USE_PRODUCTION = false`
- **Remote testing:** `USE_PRODUCTION = true`
- **iOS Simulator:** `USE_SIMULATOR = true`
- **Physical device:** `USE_SIMULATOR = false`

## Troubleshooting

### Deployment fails
- Check Railway build logs for Python dependency errors
- Ensure all dependencies are in `requirements.txt`

### App can't connect to Railway
- Verify Railway URL in `api.config.ts` (must include `https://`)
- Check Railway logs for server errors
- Ensure environment variables are set correctly

### SSE streaming not working
- Railway supports SSE by default
- Check that `sse-starlette` is in `requirements.txt`
- Verify CORS settings allow your mobile app

## Cost

- Railway free tier: 500 hours/month ($0)
- After free tier: ~$5-10/month
- Automatic scaling based on usage

## Monitoring

Railway dashboard provides:
- Real-time logs
- CPU/Memory usage
- Request metrics
- Deployment history

## Next Steps

Once deployed and tested:
1. Share the mobile app via Expo Go QR code
2. Testers can scan QR code and test from anywhere
3. Monitor Railway logs for any production issues
