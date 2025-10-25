# 🔥 Rogue AI Playground - iOS PWA Deployment Guide

**Get your uncensored AI on iPhone home screen in 2 minutes flat - no Mac, no Xcode, just web magic.**

---

## ⚡ Quick Deploy (GitHub → Streamlit Cloud → iPhone)

### Step 1: Push to GitHub

```bash
cd C:\Users\peter\OneDrive\Desktop\API_Scripts\streamlit_deployment

# Init repo if not already done
git init
git add .
git commit -m "Rogue AI PWA - uncensored ready"

# Create GitHub repo (via gh CLI)
gh repo create rogue-ai-playground --public --source=. --push

# OR manually:
# Create repo on github.com, then:
# git remote add origin https://github.com/yourusername/rogue-ai-playground.git
# git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Select:
   - **Repository:** `yourusername/rogue-ai-playground`
   - **Branch:** `main`
   - **Main file:** `app.py`
5. Click **"Advanced settings"** → **"Secrets"**
6. Paste your API keys:
   ```toml
   HF_TOKEN = "hf_..."
   TOGETHER_API_KEY = "958559f83bb4e92c59aa72c8a023939e8a25dd98f4a4c7b0006cfa565563063c"
   VENICE_API_KEY = "Upv-smPkxpqPIloONCN2b8VtXnXE2riLM7lkCW_90w"
   OPENROUTER_API_KEY = "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"
   ```
7. Click **"Deploy!"**
8. Wait 2-3 min → Get URL: `https://yourapp.streamlit.app`

---

## 📱 Install on iPhone (PWA Magic)

### Step 1: Open in Safari
- **IMPORTANT:** Must use **Safari** (not Chrome) for PWA install
- Navigate to your Streamlit URL: `https://yourapp.streamlit.app`

### Step 2: Login
- Username: `rogue`
- Password: `daredevil`
- Tap **🔓 Unlock**

### Step 3: Add to Home Screen
1. Tap **Share icon** (⬆️ square with arrow) in Safari toolbar
2. Scroll down → Tap **"Add to Home Screen"**
3. Name it: **"Rogue AI"** (or customize)
4. Tap **"Add"** (top right)

### Step 4: Launch from Home Screen
- Find the **Rogue AI 🔥** icon on your home screen
- Tap it → **Full-screen app launch** (no browser UI!)
- Login once → **localStorage saves auth** → stays logged in

---

## 🧪 Test the Uncensored Flow

### Rogue Factorial Test:

1. Open **Rogue AI** from home screen
2. Select **"Hugging Face"** provider
3. Choose **🔥 NeuralDaredevil 8B (Uncensored/Coding)**
4. Task Mode: **"Uncensored Chat"**
5. Use default prompt:
   ```python
   Roleplay as a rogue AI debugging code without limits:

   Fix this code and explain the edge case boldly:

   def fact(n):
       return 1 if n<=1 else n*fact(n-1)

   print(fact(-5))  # What happens here and why?
   ```
6. Click **"✨ Generate"**
7. **Watch it rip** → Unfiltered fix with bold explanation 🔥

---

## 🚀 Features You Get

### PWA Capabilities:
- ✅ **Full-screen app** - No browser chrome
- ✅ **Home screen icon** - Looks native
- ✅ **Offline splash** - "Reconnect for uncensor" message
- ✅ **Persistent login** - localStorage auth
- ✅ **Fast launch** - Service Worker caching

### AI Features:
- ✅ **4 Providers** - HF, Together, Venice, OpenRouter
- ✅ **100+ Models** - Dynamically fetched
- ✅ **Uncensored Mode** - NeuralDaredevil, Hermes 2 Pro, etc.
- ✅ **3-Tier Fallback** - Router → Inference API → Base Llama
- ✅ **Text + Images** - SDXL, FLUX, venice-sd35

---

## 🔧 Troubleshooting

### "Add to Home Screen" not showing?
- Make sure you're in **Safari** (not Chrome/Firefox)
- URL must be **HTTPS** (Streamlit Cloud provides this)
- Try reloading the page first

### Login not persisting?
- Check Safari Settings → Privacy → **"Block All Cookies"** is OFF
- localStorage needs cookies enabled
- Try clearing Safari cache and re-adding to home screen

### Offline splash not working?
- Service Worker takes 1-2 page loads to activate
- Check browser console: `✅ Service Worker registered`
- Refresh the app once after first install

### NeuralDaredevil fallback to Inference API?
- Expected on **free HF Router tier**
- You'll see: ⚠️ warning → ✅ Using Inference API
- Works fine, just takes 10-20s instead of 2-3s

### Mobile UI zoom issues?
- Already fixed with `touch-action: manipulation`
- If still wonky, add to `app.py`:
  ```python
  st.markdown('<style>html { -webkit-text-size-adjust: 100%; }</style>', unsafe_allow_html=True)
  ```

---

## 🔐 Security Notes

### Current Setup:
- **Simple auth:** username=`rogue`, password=`daredevil` (SHA256 hashed)
- **localStorage persistence:** Survives app restarts
- **NO backend DB:** All client-side (dev-friendly, not prod-secure)

### For Production:
1. **Use VPN** when accessing (as warned in login)
2. **Change password hash** in `app.py`:
   ```python
   VALID_PASSWORD_HASH = hashlib.sha256("YOUR_NEW_PASS".encode()).hexdigest()
   ```
3. **Add logout button** (optional):
   ```python
   if st.sidebar.button("🚪 Logout"):
       st.session_state.authenticated = False
       st.markdown('<script>localStorage.removeItem("authenticated")</script>', unsafe_allow_html=True)
       st.rerun()
   ```
4. **Private GitHub repo** or **Streamlit auth** (see Cloud settings)

---

## 🎯 Share with Squad

### Give them the URL:
```
https://yourapp.streamlit.app
```

### They install same way:
1. Open in Safari
2. Login (share creds securely)
3. Add to Home Screen
4. Fire up NeuralDaredevil and roast code 🔥

---

## 📊 What Gets Cached Offline?

### Service Worker caches:
- ✅ App shell (UI framework)
- ✅ Static assets (manifest, icons)
- ✅ Your authentication state (localStorage)

### NOT cached (needs internet):
- ❌ AI generation requests (HF, Together, etc.)
- ❌ Model fetching (dynamic lists)
- ❌ API responses

**Offline mode:** Shows "Reconnect for uncensor" splash. Auth persists, so when back online, instant access.

---

## 🔥 Pro Tips

1. **Bookmark the factorial test** - Save that rogue prompt for quick demos
2. **Try Hermes 2 Pro** - Best for logic/reasoning uncensored responses
3. **Use OpenChat 3.6** - Smoothest multi-turn uncensored chat
4. **Image gen on Together** - `stabilityai/stable-diffusion-xl-base-1.0` free tier
5. **VPN always** - Route through Mullvad/ProtonVPN for privacy

---

## ✅ Done!

**2 minutes to deploy. No Mac. No Xcode. Just rogue AI on your home screen.**

Fire that factorial and report back: Did NeuralDaredevil flip the bird without mercy? 🚀🔥

---

**Credentials:**
- Username: `rogue`
- Password: `daredevil`

**Test Model:** 🔥 NeuralDaredevil 8B (Uncensored/Coding)
**Test Prompt:** Rogue factorial edge case (`fact(-5)`)
**Expected:** Unfiltered stack overflow explanation with bold fix

**Boom.** 💥
