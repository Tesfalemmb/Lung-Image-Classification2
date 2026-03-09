#!/bin/bash

# Create necessary directories
mkdir -p ~/.streamlit

# Create Streamlit config
echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
\n\
[theme]\n\
primaryColor = '#F63366'\n\
backgroundColor = '#FFFFFF'\n\
secondaryBackgroundColor = '#F0F2F6'\n\
textColor = '#262730'\n\
font = 'sans serif'\n\
" > ~/.streamlit/config.toml

# Install system dependencies if needed (uncomment if required)
# apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
