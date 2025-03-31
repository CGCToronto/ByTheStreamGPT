# ByTheStreamGPT

A fine-tuned language model based on DeepSeek-R1-Distill-Qwen-1.5B, specifically optimized for spiritual knowledge and wisdom.

## Overview

ByTheStreamGPT is a specialized language model that has been fine-tuned to provide high-quality responses in the domain of spiritual knowledge and wisdom. The model is built upon the DeepSeek-R1-Distill-Qwen-1.5B base model and has been optimized through careful training and fine-tuning processes.

## Features

- Specialized in spiritual knowledge and wisdom
- Optimized response quality
- Efficient inference with Ollama integration
- Comprehensive training metrics tracking
- Easy-to-use testing interface

## Project Structure

```
ByTheStreamGPT/
├── fine_tuning/           # Fine-tuning related files
│   ├── data/             # Training data
│   ├── models/           # Model files
│   ├── output/           # Training outputs
│   ├── prepare_data.py   # Data preparation script
│   ├── train.py          # Training script
│   └── test_model.py     # Model testing script
└── README.md             # Project documentation
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Ollama
- Other dependencies listed in requirements.txt

## Setup

1. Clone the repository:
```bash
git clone https://github.com/CGCToronto/ByTheStreamGPT.git
cd ByTheStreamGPT
```

2. Install dependencies:
```bash
pip install -r fine_tuning/requirements.txt
```

3. Install Ollama and set up the model:
```bash
# Follow Ollama installation instructions for your platform
# Then pull the model
ollama pull deepseek-coder:1.3b-base
```

## Usage

### Testing the Model

To test the fine-tuned model:

```bash
python fine_tuning/test_model.py
```

This will start an interactive session where you can input questions and receive responses from the model.

## Training

The model has been fine-tuned using carefully curated spiritual knowledge data. Training metrics and configurations are available in the fine_tuning directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DeepSeek for the base model
- The open-source community for various tools and libraries
- All contributors to this project 