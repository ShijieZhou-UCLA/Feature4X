#!/bin/bash

# Feature4X Environment Variables Setup Script
# This script helps you set up required environment variables for Feature4X

echo "🌟 Feature4X Environment Variables Setup"
echo "========================================"
echo ""

# Check if variables are already set
echo "📋 Current Environment Status:"
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ OPENAI_API_KEY is already set"
else
    echo "❌ OPENAI_API_KEY is not set"
fi

if [ -n "$HF_TOKEN" ]; then
    echo "✅ HF_TOKEN is already set"
else
    echo "❌ HF_TOKEN is not set"
fi

echo ""
echo "🔧 Setup Options:"
echo "1. 🔄 Update environment variables for current session only"
echo "2. 💾 Add environment variables to ~/.bashrc (persistent)"
echo "3. 📖 Show manual setup instructions"
echo "4. 🚪 Exit"
echo ""

read -p "Choose an option (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🔄 Setting up environment variables for current session..."
        
        if [ -z "$OPENAI_API_KEY" ]; then
            echo ""
            echo "🔐 OpenAI API Key Setup:"
            echo "Get your API key at: https://platform.openai.com/account/api-keys"
            read -p "Enter your OpenAI API key: " openai_key
            if [ -n "$openai_key" ]; then
                export OPENAI_API_KEY="$openai_key"
                echo "✅ OPENAI_API_KEY set for current session"
            else
                echo "⚠️  Skipped OpenAI API key setup"
            fi
        fi
        
        if [ -z "$HF_TOKEN" ]; then
            echo ""
            echo "🤗 Hugging Face Token Setup:"
            echo "Get your token at: https://huggingface.co/settings/tokens"
            read -p "Enter your Hugging Face token: " hf_token
            if [ -n "$hf_token" ]; then
                export HF_TOKEN="$hf_token"
                echo "✅ HF_TOKEN set for current session"
            else
                echo "⚠️  Skipped Hugging Face token setup"
            fi
        fi
        
        echo ""
        echo "✅ Environment variables are now set for this session!"
        echo "💡 Run this script again and choose option 2 to make them persistent."
        ;;
        
    2)
        echo ""
        echo "💾 Adding environment variables to ~/.bashrc..."
        
        # Backup existing bashrc
        cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
        echo "📁 Backup created: ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)"
        
        if [ -z "$OPENAI_API_KEY" ] && ! grep -q "OPENAI_API_KEY" ~/.bashrc; then
            echo ""
            echo "🔐 OpenAI API Key Setup:"
            echo "Get your API key at: https://platform.openai.com/account/api-keys"
            read -p "Enter your OpenAI API key: " openai_key
            if [ -n "$openai_key" ]; then
                echo "" >> ~/.bashrc
                echo "# Feature4X - OpenAI API Key" >> ~/.bashrc
                echo "export OPENAI_API_KEY=\"$openai_key\"" >> ~/.bashrc
                export OPENAI_API_KEY="$openai_key"
                echo "✅ OPENAI_API_KEY added to ~/.bashrc"
            else
                echo "⚠️  Skipped OpenAI API key setup"
            fi
        fi
        
        if [ -z "$HF_TOKEN" ] && ! grep -q "HF_TOKEN" ~/.bashrc; then
            echo ""
            echo "🤗 Hugging Face Token Setup:"
            echo "Get your token at: https://huggingface.co/settings/tokens"
            read -p "Enter your Hugging Face token: " hf_token
            if [ -n "$hf_token" ]; then
                echo "" >> ~/.bashrc
                echo "# Feature4X - Hugging Face Token" >> ~/.bashrc
                echo "export HF_TOKEN=\"$hf_token\"" >> ~/.bashrc
                export HF_TOKEN="$hf_token"
                echo "✅ HF_TOKEN added to ~/.bashrc"
            else
                echo "⚠️  Skipped Hugging Face token setup"
            fi
        fi
        
        echo ""
        echo "✅ Environment variables added to ~/.bashrc!"
        echo "🔄 Run 'source ~/.bashrc' or restart your terminal to activate them."
        echo "💡 The variables are also set for the current session."
        ;;
        
    3)
        echo ""
        echo "📖 Manual Setup Instructions:"
        echo "=============================="
        echo ""
        echo "🔐 OpenAI API Key:"
        echo "1. Get your API key at: https://platform.openai.com/account/api-keys"
        echo "2. Set environment variable:"
        echo "   export OPENAI_API_KEY=\"your_api_key_here\""
        echo "3. Add to ~/.bashrc for persistence:"
        echo "   echo 'export OPENAI_API_KEY=\"your_api_key_here\"' >> ~/.bashrc"
        echo ""
        echo "🤗 Hugging Face Token:"
        echo "1. Get your token at: https://huggingface.co/settings/tokens"
        echo "2. Set environment variable:"
        echo "   export HF_TOKEN=\"your_token_here\""
        echo "3. Add to ~/.bashrc for persistence:"
        echo "   echo 'export HF_TOKEN=\"your_token_here\"' >> ~/.bashrc"
        echo ""
        echo "💡 After adding to ~/.bashrc, run: source ~/.bashrc"
        ;;
        
    4)
        echo "👋 Goodbye!"
        exit 0
        ;;
        
    *)
        echo "❌ Invalid option. Please choose 1-4."
        ;;
esac

echo ""
echo "🎉 Setup complete! You can now use Feature4X without entering API keys every time."
echo "📝 To verify your setup, run: python feature4x_interactive.py"