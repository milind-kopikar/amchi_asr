# Railway Deployment Guide for Konkani Dictionary

## 🚂 Step-by-Step Railway Setup

### 1. Prepare Repository for Railway
```bash
# Make sure you're in the project directory
cd konkani_dictionary

# Initialize git if not already done
git init
git add .
git commit -m "Initial commit - Konkani Dictionary with 4,381 entries"

# Push to GitHub
git remote add origin https://github.com/moksh-kopikar/konkani-dictionary.git
git push -u origin main
```

### 2. Railway Account Setup
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub account
3. Connect your GitHub repositories

### 3. Create Railway Project
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your `konkani-dictionary` repository
4. Railway will detect the Node.js app automatically

### 4. Add PostgreSQL Database
1. In your Railway project dashboard
2. Click "New Service"
3. Select "Database" → "PostgreSQL"
4. Railway will provision a PostgreSQL instance

### 5. Configure Environment Variables
In Railway dashboard, add these variables:
```
NODE_ENV=production
FRONTEND_URL=https://moksh-kopikar.github.io
ALLOWED_ORIGINS=https://moksh-kopikar.github.io,https://your-railway-domain.railway.app
```

### 6. Deploy Database
```bash
# Railway will provide DATABASE_URL automatically
# Run migration script after deployment
npm run migrate:railway
```

### 7. Test Deployment
- Railway will provide a URL like: `https://your-app-name.railway.app`
- Test endpoints:
  - `https://your-app-name.railway.app/health`
  - `https://your-app-name.railway.app/api/stats`
  - `https://your-app-name.railway.app/api/dictionary?limit=5`

## 🔧 Environment Setup Commands
```bash
# Install Railway CLI (optional)
npm install -g @railway/cli

# Login to Railway
railway login

# Link to existing project
railway link [project-id]

# Deploy manually
railway up
```

## 📊 Expected Results
- **Backend API:** Hosted on Railway with PostgreSQL
- **Database:** 4,381 Konkani entries migrated
- **Endpoints:** All API routes accessible via HTTPS
- **Performance:** Auto-scaling based on usage