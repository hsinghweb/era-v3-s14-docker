---
title: SmolLM2 135 Text Generator
emoji: 🐢
colorFrom: green
colorTo: green
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Text generation using SmolLM2-135 model
---

# SmolLM2-135 Text Generator

This is a Streamlit-based text generation application using a fine-tuned SmolLM2-135 model. The application allows users to:

- Input custom prompts
- Control the length of generated text
- Generate multiple text sequences
- View token information

## Features

- Interactive text input
- Adjustable text generation length
- Multiple sequence generation
- Real-time text generation
- Token information display

## Usage

1. Enter your prompt in the text area
2. Adjust the length of text to be generated
3. Select the number of sequences to generate
4. Click "Generate" to create text

## Technical Details

The application uses:
- SmolLM2-135 model architecture
- Tiktoken tokenizer
- PyTorch for model inference
- Streamlit for the user interface
