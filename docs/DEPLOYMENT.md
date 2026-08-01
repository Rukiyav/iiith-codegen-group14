# Deployment Guide — CodeGen Group14

This document explains how to deploy the **CodeGen Group14 AI System** locally, in Docker containers, and on cloud platforms.

---

## 1. Local & Network Deployment (Quickstart & Live Demo)

### Option A: Interactive Streamlit Web UI
To run the web interface accessible on your local network (for capstone demonstrations):

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
* **Local Access**: `http://localhost:8501`
* **Network Access**: `http://<your-machine-ip>:8501`

---

### Option B: FastAPI Production REST Server (Uvicorn)
To run the REST API server with automatic OpenAPI documentation:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
```
* **API Documentation**: `http://localhost:8000/docs`
* **Health Check**: `http://localhost:8000/health`
* **Endpoints**:
  * `POST /generate/code` — NL to Code Generation
  * `POST /docs/generate` — Code to Docstring Generation
  * `POST /translate` — Programming Language Translation
  * `POST /rag/index` — Project Directory Indexing

---

## 2. Docker Container Deployment

You can package and deploy the application into an isolated Docker container.

### Step 1: Create `Dockerfile` in Repository Root

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit and FastAPI ports
EXPOSE 8501 8000

# Default entrypoint (Streamlit Web UI)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Build & Run Container

```bash
# Build the Docker image
docker build -t codegen-group14 .

# Run container (Streamlit Web UI on port 8501)
docker run -d -p 8501:8501 --name codegen-app codegen-group14

# Or run container (FastAPI REST Server on port 8000)
docker run -d -p 8000:8000 --name codegen-api codegen-group14 uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 3. Free Cloud Deployment Options

### A. HuggingFace Spaces (Recommended for Streamlit Web UI)
1. Go to [HuggingFace Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Select **Streamlit** as the SDK and choose **CPU Basic** (Free tier).
3. Connect your GitHub repository `Rukiyav/iiith-codegen-group14`.
4. HuggingFace automatically reads `requirements.txt` and runs `app.py` live on the web!

### B. Render / Railway / AWS EC2
1. Push your code with `Dockerfile` to GitHub.
2. Link your repository to **Render** or **Railway**.
3. Set environment variable `PORT=8501` and build command `docker build`.
4. Render/Railway will automatically build and deploy your container with an HTTPS URL.
