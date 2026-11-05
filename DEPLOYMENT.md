# 🚀 VisionGuard AI - Production Deployment Guide

VisionGuard AI can be deployed across multiple environments:
1. **Streamlit Community Cloud (Free & Fastest)** - Public cloud deployment with free MongoDB Atlas.
2. **Docker / Docker Compose (Production Ready)** - Deploy anywhere (AWS EC2, GCP, DigitalOcean, or Local Security Workstation) with an isolated MongoDB container.
3. **Render / Railway (PaaS)** - Containerized cloud app hosting.

---

## Option 1: Streamlit Community Cloud (Free 1-Click Cloud Deployment)

### Step 1: Initialize Git and Push to GitHub
Open your terminal in `D:\VisionGuard-AI` and run:
```bash
git init
git add .
git commit -m "feat: VisionGuard AI full surveillance command center"
git branch -M main
git remote add origin https://github.com/<your-username>/VisionGuard-AI.git
git push -u origin main
```

### Step 2: Set Up Free MongoDB Atlas (Cloud Database)
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) and create a free M0 cluster.
2. Under **Database Access**, create a user (e.g., `admin`) and password.
3. Under **Network Access**, add `0.0.0.0/0` (Allow access from anywhere).
4. Click **Connect** -> **Drivers (Python)** and copy your connection string:
   ```text
   mongodb+srv://admin:<password>@cluster0.xyz.mongodb.net/?retryWrites=true&w=majority
   ```

### Step 3: Deploy on Streamlit Cloud
1. Sign in to [share.streamlit.io](https://share.streamlit.io/).
2. Click **New App** and select your GitHub repository (`VisionGuard-AI`).
3. Set:
   - **Main file path**: `app.py`
4. Expand **Advanced settings...** -> **Secrets** and add:
   ```toml
   MONGODB_URI = "mongodb+srv://admin:<password>@cluster0.xyz.mongodb.net/?retryWrites=true&w=majority"
   MONGODB_DATABASE = "VisionGuardAI"
   DEFAULT_CAMERA_ID = "CAM_01"
   ```
5. Click **Deploy!**

---

## Option 2: Docker & Docker Compose (Recommended for Production & On-Premise)

This bundles VisionGuard AI, all OpenCV system libraries, and an isolated MongoDB container with a single command.

### Step 1: Start Containers
From the `VisionGuard-AI` directory:
```bash
docker compose up -d --build
```

### Step 2: Access the Dashboard
- **Dashboard URL**: `http://localhost:8501` (or `http://<server-ip>:8501`)
- **MongoDB Database**: Running on port `27017`

### Step 3: Useful Commands
```bash
# View live application logs
docker compose logs -f visionguard

# Stop containers
docker compose down

# Restart containers
docker compose restart
```

---

## Option 3: Deploy to Cloud VPS (AWS EC2 / DigitalOcean / GCP)

1. Provision an Ubuntu 22.04 LTS instance (t3.medium or larger recommended).
2. Install Docker and Git:
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
   sudo systemctl enable --now docker
   ```
3. Clone your repository:
   ```bash
   git clone https://github.com/<your-username>/VisionGuard-AI.git
   cd VisionGuard-AI
   ```
4. Start with Docker Compose:
   ```bash
   sudo docker compose up -d --build
   ```
5. Ensure your Cloud Firewall / Security Group allows inbound traffic on port `8501`.

---

## Hardware Acceleration Notes (Optional)
- For GPU acceleration (NVIDIA CUDA): Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and add `deploy.resources.reservations.devices` with GPU driver capabilities into `docker-compose.yml`.

