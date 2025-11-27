#!/usr/bin/env python3
"""
Integrated Web Chatbot System Launcher
=====================================

This script launches the complete integrated system that includes:
1. Deep web scraping (deep_scraper.py functionality)
2. Basic semantic chatbot (basic_chatbot.py functionality) 
3. Gemini AI chatbot
4. Flask web interface

Usage:
    python run_system.py

The system will:
- Automatically start web scraping in the background
- Load the Flask web server
- Provide both chatbot interfaces through the web UI
"""

import os
import sys
import time
import threading
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'flask',
        'sentence_transformers', 
        'google-generativeai',
        'beautifulsoup4',
        'requests',
        'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_files():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'deep_scraper.py', 
        'basic_chatbot.py',
        'templates/index.html'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files are present")
    return True

def check_env_file():
    """Check if .env file exists and has API key"""
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("💡 Create a .env file with your Google API key:")
        print("   GOOGLE_API_KEY=your_api_key_here")
        print("   Note: The system will work without it, but Gemini responses won't be available")
        return False
    
    # Check if API key is in .env file
    try:
        with open(env_file, 'r') as f:
            content = f.read()
            if 'GOOGLE_API_KEY=' in content:
                print("✅ .env file found with API key")
                return True
            else:
                print("⚠️  .env file found but no GOOGLE_API_KEY detected")
                return False
    except Exception as e:
        print(f"⚠️  Error reading .env file: {e}")
        return False

def main():
    """Main launcher function"""
    print("🚀 Integrated Web Chatbot System Launcher")
    print("=" * 50)
    
    # Check system requirements
    print("\n🔍 Checking system requirements...")
    if not check_dependencies():
        sys.exit(1)
    
    if not check_files():
        sys.exit(1)
    
    check_env_file()
    
    print("\n🎯 System ready to launch!")
    print("\n📋 What will happen:")
    print("   1. 🕷️  Automatic web scraping will start in background")
    print("   2. 🌐 Flask web server will start")
    print("   3. 💬 Both chatbot interfaces will be available")
    print("   4. 🔄 You can manually trigger scraping anytime")
    
    print("\n⏳ Starting in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    print("\n🚀 Launching integrated system...")
    
    # Import and run the Flask app
    try:
        from app import app
        print("✅ Successfully imported Flask app")
        print("🌐 Starting web server...")
        print("📱 Open your browser to: http://localhost:5000")
        print("⏹️  Press Ctrl+C to stop the system")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"❌ Error importing Flask app: {e}")
        print("💡 Make sure app.py is in the current directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 System stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
