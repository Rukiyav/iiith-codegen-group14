# Step-by-Step Guide: Deploying to HuggingFace Spaces (Free)

This guide provides exact, step-by-step instructions to deploy the **CodeGen Group14 Web Application & APIs** on **HuggingFace Spaces** for free.

---

## Step 1: Create a HuggingFace Account & Space

1. Go to **[HuggingFace.co](https://huggingface.co/)** and log in (or create a free account).
2. Click on your profile picture at the top right and select **New Space** (or navigate to [huggingface.co/new-space](https://huggingface.co/new-space)).
3. Fill in the Space configuration:
   * **Space Name**: `codegen-group14`
   * **License**: `MIT`
   * **Select the Space SDK**: Choose **Streamlit**
   * **Space Hardware**: Choose **CPU Basic • 2 vCPU • 16 GB RAM (Free)**
4. Click **Create Space**.

---

## Step 2: Prepare Repository Files

HuggingFace Spaces requires a YAML header in the `README.md` file to identify the entry point.

### Add Frontmatter Header to `README.md`:
Make sure the very top of your project `README.md` contains this YAML header:

```yaml
---
title: CodeGen Group14 AI System
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---
```

---

## Step 3: Push Code to HuggingFace Spaces

You can push your repository directly to HuggingFace using Git.

Run the following commands in your local terminal:

```bash
# 1. Add HuggingFace Spaces as a git remote
git remote add hf https://huggingface.co/spaces/<YOUR-HF-USERNAME>/codegen-group14

# 2. Push your main branch to HuggingFace
git push hf rajat-initial-commit:main
```

*(Note: Replace `<YOUR-HF-USERNAME>` with your actual HuggingFace username).*

---

## Step 4: Verify Deployment

1. Return to your Space page on HuggingFace: `https://huggingface.co/spaces/<YOUR-HF-USERNAME>/codegen-group14`
2. HuggingFace will display **"Building..."** while installing dependencies from `requirements.txt`.
3. After ~2–3 minutes, the status will change to **"Running"**.
4. Your interactive 3-tab Web UI will be live with a public URL!

---

## Optional: Expose FastAPI REST Endpoints on HF Spaces

If you also want to expose the **FastAPI OpenAPI REST documentation** on HuggingFace:

1. In your HuggingFace Space **Settings**, change the Space SDK to **Docker**.
2. Push a `Dockerfile` with this entry point:
   ```dockerfile
   CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
   ```
3. Your FastAPI endpoints will be available live at:
   `https://<YOUR-HF-USERNAME>-codegen-group14.hf.space/docs`
