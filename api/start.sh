#!/bin/bash

# Startup script for Hierarchical Data Simulator API

echo "🎲 Starting Hierarchical Data Simulator API..."
echo "================================================"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.10 or higher."
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if virtual environment should be created
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements_api.txt

# Install hierarchical-simulator library
echo "📦 Installing hierarchical-simulator library..."
cd ..
pip install -q -e .
cd api

# Check if installation was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed successfully"
echo ""
echo "🚀 Starting API server..."
echo "================================================"
echo "📍 Web Interface: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🏥 Health Check: http://localhost:8000/api/v1/health"
echo "================================================"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python main.py
