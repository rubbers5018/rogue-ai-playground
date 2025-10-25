# 🚀 AI Multi-Provider Playground

Production-ready Streamlit app integrating **Together.ai**, **Venice AI**, and **OpenRouter** for text and image generation with 100+ AI models. Optimized for **Streamlit Cloud** deployment with secrets management, 30-minute caching, and session fallback.

---

## ✨ Features

- **3 AI Providers**: Together.ai, Venice AI, OpenRouter (text-only)
- **100+ Models**: Dynamically fetched and cached (30min TTL)
- **Text Generation**: Chat, coding assistance with system prompts
- **Image Generation**: Together.ai (SDXL, FLUX), Venice AI (venice-sd35)
- **Smart Caching**: 30-min cache + session state fallback on API failures
- **Secrets-Only**: Streamlit secrets (no exposed keys in UI)
- **Task Modes**: Coding Assistant, General Chat, Image Gen
- **Robust Errors**: 429 rate limits, timeouts, HTTP errors handled
- **2025 Endpoints**: Corrected API paths for all providers

---

## 📋 Quick Start

### Prerequisites

- Python 3.12+
- API keys from:
  - [Together.ai](https://api.together.xyz/signup) (free tier available)
  - [Venice AI](https://venice.ai) (free tier available)
  - [OpenRouter](https://openrouter.ai) (pay-as-you-go)

### Local Setup

1. **Clone/Download this repo**
   ```bash
   cd ai-playground
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create secrets file**
   ```bash
   mkdir .streamlit
   ```

   Create `.streamlit/secrets.toml`:
   ```toml
   TOGETHER_API_KEY = "your_together_key_here"
   VENICE_API_KEY = "your_venice_key_here"
   OPENROUTER_API_KEY = "your_openrouter_key_here"
   ```

   **Or use the provided template:**
   ```bash
   cp .streamlit/secrets.toml.template .streamlit/secrets.toml
   # Edit secrets.toml with your keys
   ```

4. **Run locally**
   ```bash
   streamlit run app.py
   ```

   App will open at `http://localhost:8501`

---

## 🌐 Deploy to Streamlit Cloud

### Step 1: Push to GitHub

**Option A: Using GitHub CLI (`gh`)**
```bash
# From repo root
git init
git add .
git commit -m "Initial AI playground"

# Create repo and push (gh CLI required)
gh repo create ai-playground --public --source=. --push
```

**Option B: Manual GitHub**
```bash
git init
git add .
git commit -m "Initial AI playground"

# Create repo on github.com, then:
git remote add origin https://github.com/yourusername/ai-playground.git
git branch -M main
git push -u origin main
```

**Or use the provided script:**
```bash
bash deploy.sh
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select:
   - **Repository**: `yourusername/ai-playground`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Advanced settings"** → **"Secrets"**
6. Paste your secrets (TOML format):
   ```toml
   TOGETHER_API_KEY = "sk-..."
   VENICE_API_KEY = "sk-..."
   OPENROUTER_API_KEY = "sk-..."
   ```
7. Click **"Deploy!"**

Your app will be live at: `https://yourusername-ai-playground-xyz.streamlit.app`

---

## 🎮 Usage Guide

### Text Generation

1. Select **Provider** (Together.ai, Venice AI, OpenRouter)
2. Choose **Model** (e.g., `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free`)
3. Pick **Task Mode**:
   - **Coding Assistant**: Python debugging, algorithm help
   - **General Chat**: Questions, explanations
4. Enter **Prompt**
5. Adjust **Parameters**:
   - Temperature (0.0-2.0): Lower = focused, Higher = creative
   - Max Tokens (512-4096): Output length
   - Top P (0.0-1.0): Nucleus sampling
6. Click **"✨ Generate"**
7. Output displays in formatted markdown/code blocks
8. Download text via **"💾 Download Text"**

### Image Generation

1. Select **Provider** (Together.ai or Venice AI only)
2. Choose **Image Model**:
   - Together.ai: `stabilityai/stable-diffusion-xl-base-1.0`, `black-forest-labs/FLUX.1-schnell`
   - Venice AI: `venice-sd35` (via chat endpoint)
3. Check **"🎨 Generate Image"**
4. Enter detailed **Prompt** (e.g., "A cyberpunk cityscape at night, neon lights, 8k")
5. Adjust **Dimensions** (Width/Height: 512x512 recommended)
6. Click **"✨ Generate"** (takes 30-120s)
7. Download via **"💾 Download Image"**

**Note:** OpenRouter does not support image generation (text-only).

### Cache Management

- **Models cached 30 minutes** (reduces API calls)
- Click **"🔄 Refresh Model Cache"** in sidebar to force update
- **Session fallback**: If API fails, uses last successful fetch
- Cache applies per provider (max 3 entries)

---

## 🔧 2025 API Endpoints (Corrected)

### Together.ai
- **Base:** `https://api.together.xyz`
- **Models:** `GET /v1/models`
- **Chat:** `POST /v1/chat/completions`
- **Images:** `POST /v1/images/generations`
- **Auth:** `Bearer {TOGETHER_API_KEY}`

### Venice AI
- **Base:** `https://api.venice.ai`
- **Models:** `GET /models` (NOT `/api/v1/models`)
- **Chat:** `POST /api/v1/chat/completions`
- **Images:** Via chat endpoint with `model="venice-sd35"` (no `/images`)
- **Auth:** `Bearer {VENICE_API_KEY}`

### OpenRouter
- **Base:** `https://openrouter.ai/api/v1`
- **Models:** `GET /api/v1/models` (NOT `/models`)
- **Chat:** `POST /chat/completions`
- **Images:** ❌ Not supported (text-only)
- **Auth:** `Bearer {OPENROUTER_API_KEY}` + headers:
  - `HTTP-Referer: https://github.com/yourusername/ai-playground`
  - `X-Title: AI Playground`

---

## 🛠️ Troubleshooting

### "No API keys configured!"
- **Local:** Create `.streamlit/secrets.toml` with your keys
- **Cloud:** Add secrets in Streamlit Cloud Settings → Secrets

### "No models available"
- Check API key is correct (no extra spaces)
- Try **"🔄 Refresh Model Cache"** button
- Check internet connection
- Verify API key hasn't expired

### "Rate limit exceeded (429)"
- Wait 1-5 minutes before retrying
- Free tiers have hourly limits
- Consider upgrading plan

### "Request timed out"
- Model may be overloaded—try different model
- Reduce `max_tokens` (lower = faster)
- Image generation takes 30-120s (normal)

### Images not generating
- **Check provider**: OpenRouter doesn't support images
- **Check model**: Must be image-capable (SDXL, FLUX, venice-sd35)
- **Venice AI**: Uses chat endpoint, not dedicated `/images`
- Try 512x512 dimensions (most compatible)

### Models not refreshing
- Click **"🔄 Refresh Model Cache"** to clear 30-min cache
- Session fallback shows "Using cached models from previous session"
- Check API key if cache always fails

### Code blocks not rendering
- Coding Assistant mode uses `st.code()` for proper formatting
- If markdown not rendering, check for special characters

---

## 📁 Project Structure

```
ai-playground/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── deploy.sh                       # Deployment helper script
├── .gitignore                      # Git ignore rules
└── .streamlit/
    ├── secrets.toml                # API keys (NEVER commit!)
    └── secrets.toml.template       # Template for secrets
```

---

## 🔒 Security Notes

- **Never commit** `.streamlit/secrets.toml` to Git (already in `.gitignore`)
- Use **Streamlit Cloud Secrets** for production (encrypted)
- Rotate API keys if accidentally exposed
- OpenRouter requires `HTTP-Referer` header (set to your repo URL)

---

## 📊 Model Recommendations

### Best Free Models

**Text Generation:**
- Together.ai: `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free`
- Venice AI: `llama-3.3-70b`
- OpenRouter: `google/gemini-flash-1.5-8b` (free tier)

**Coding:**
- Together.ai: `Qwen/Qwen2.5-Coder-32B-Instruct`
- OpenRouter: `anthropic/claude-3.5-sonnet` (paid but best)

**Image Generation:**
- Together.ai: `stabilityai/stable-diffusion-xl-base-1.0` (free)
- Together.ai: `black-forest-labs/FLUX.1-schnell` (fast)
- Venice AI: `venice-sd35` (via chat endpoint)

---

## 🤝 Contributing

Issues and PRs welcome! To contribute:
1. Fork this repo
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

MIT License - feel free to use for personal or commercial projects.

---

## 🔗 Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [Streamlit Cloud Deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [Streamlit Secrets Management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Together.ai API Docs](https://docs.together.ai)
- [Venice AI Docs](https://docs.venice.ai)
- [OpenRouter API Docs](https://openrouter.ai/docs)

---

## 🎯 Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Run on custom port
streamlit run app.py --server.port 8502

# Deploy to GitHub (via script)
bash deploy.sh

# Clear Streamlit cache (in-app or CLI)
# In app: Click "🔄 Refresh Model Cache"
# Or restart server
```

---

**Built with ❤️ | 2025 Endpoints | Streamlit Cloud Ready**
