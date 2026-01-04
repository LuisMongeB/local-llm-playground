# LLM Local Playground

A lightweight framework for testing and evaluating local Large Language Models.

## Overview

This project provides a modular framework for loading, running, and evaluating various local LLM models on your machine. It supports different model architectures including causal language models (like Qwen3), encoder-decoder models (like Google T5Gemma), and RAG models (like Apple CLaRa).

## Features

- **Model Loading**: Simple utilities to load models from Hugging Face Hub
- **Device Optimization**: Automatic detection and optimization for CUDA, Apple Silicon (MPS), and CPU
- **Evaluation Framework**: Run models on benchmark datasets and grade outputs using OpenAI's GPT-4 as a judge
- **Memory Management**: Efficient resource cleanup and management
- **Multiple Architectures**: Support for causal LM, seq2seq, and RAG models

## Installation

This project uses [uv](https://github.com/astral-sh/uv) as the package manager.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Configuration

Create a `.env` file in the project root with the following:

```
HUGGINGFACE_TOKEN=your_hf_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

The project includes several example experiments in the `experiments/` directory:

```bash
# Run a simple smoke test with TinyLlama
uv run experiments/01_smoke_test.py

# Test Qwen3 models
uv run experiments/02_qwen3_smallest.py
uv run experiments/03_qwen3_4b_instruct.py

# Test Apple CLaRa with RAG
uv run experiments/04_apple_clara.py

# Test Google T5Gemma
uv run experiments/05_t5gemma_test.py
```

## Project Structure

- `src/llm_playground/` - Core package code
  - `core.py` - Device detection and memory management
  - `loader.py` - Model loading utilities
  - `evaluate.py` - Generation and evaluation logic
  - `judge.py` - Grading interfaces
- `experiments/` - Example experiment scripts
- `data/` - Evaluation datasets

## Requirements

- Python 3.10+
- PyTorch 2.9+
- Transformers 4.57+
- Optional: CUDA-capable GPU or Apple Silicon for optimized performance

## License

MIT License - see [LICENSE](LICENSE) for details.
