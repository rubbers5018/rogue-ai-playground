#!/bin/bash
# Streamlit Cloud Deployment Helper
# Automates GitHub repo creation and push

set -e  # Exit on error

echo "🚀 AI Multi-Provider Playground - Deployment Script"
echo "=================================================="
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "📦 Initializing Git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git repository already initialized"
fi

# Check if files are committed
if [ -z "$(git status --porcelain)" ]; then
    echo "✅ No changes to commit"
else
    echo "📝 Staging files..."
    git add .
    echo ""
    echo "💬 Enter commit message (or press Enter for default):"
    read -r commit_msg
    commit_msg=${commit_msg:-"Initial AI playground deployment"}

    git commit -m "$commit_msg"
    echo "✅ Changes committed"
fi

# Check if GitHub CLI is installed
if command -v gh &> /dev/null; then
    echo ""
    echo "🔍 GitHub CLI detected"
    echo "📦 Creating GitHub repository..."
    echo ""
    echo "Enter repository name (default: ai-playground):"
    read -r repo_name
    repo_name=${repo_name:-ai-playground}

    echo "Make repository public? (y/n, default: y):"
    read -r is_public
    is_public=${is_public:-y}

    if [ "$is_public" = "y" ]; then
        visibility="--public"
    else
        visibility="--private"
    fi

    # Create repo and push
    gh repo create "$repo_name" $visibility --source=. --push

    echo ""
    echo "✅ Repository created and pushed!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Go to https://share.streamlit.io"
    echo "2. Click 'New app'"
    echo "3. Select your GitHub repo: $(gh repo view --json nameWithOwner -q .nameWithOwner)"
    echo "4. Set main file: app.py"
    echo "5. Add secrets in Settings → Secrets:"
    echo ""
    echo "   TOGETHER_API_KEY = \"your_key\""
    echo "   VENICE_API_KEY = \"your_key\""
    echo "   OPENROUTER_API_KEY = \"your_key\""
    echo ""
    echo "6. Click Deploy!"

else
    echo ""
    echo "⚠️  GitHub CLI not found"
    echo "📖 Manual deployment instructions:"
    echo ""
    echo "1. Create GitHub repo:"
    echo "   - Go to https://github.com/new"
    echo "   - Create repo named 'ai-playground'"
    echo "   - Choose public/private"
    echo ""
    echo "2. Push to GitHub:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/ai-playground.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    echo "3. Deploy on Streamlit Cloud:"
    echo "   - Go to https://share.streamlit.io"
    echo "   - Click 'New app'"
    echo "   - Connect to your repo"
    echo "   - Add secrets (see README.md)"
    echo ""
    echo "💡 Install GitHub CLI for automated deployment:"
    echo "   https://cli.github.com"
fi

echo ""
echo "🎉 Deployment script complete!"
