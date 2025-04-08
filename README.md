# ByTheStream Magazine Model Fine-tuning Project

## Project Overview
This project uses the DeepSeek-R1-Distill-Qwen-1.5B model and fine-tunes it using the LoRA method on ByTheStream magazine content to generate relevant responses about the magazine's content.

## Data Preparation

### 1. Data Collection
1. Collect article content from the ByTheStream magazine website
2. Save articles in JSON format with the following fields:
   ```json
   {
     "title": "Article Title",
     "author": "Author",
     "volume": "Issue Number",
     "content": "Article Content",
     "date": "Publication Date"
   }
   ```

### 2. Data Preprocessing
We provide two data preprocessing scripts:

#### Standard Data Preparation
```bash
python prepare_data.py
```
This script:
- Cleans HTML tags
- Standardizes text format
- Generates Q&A pairs for training
- Creates training and validation sets

#### Optimized Data Preparation
```bash
python prepare_data_small.py
```
This optimized script:
- Processes only the first 10 volumes of content (reducing data size by ~50%)
- Implements advanced data augmentation techniques:
  - Synonym replacement with protected keywords
  - Sentence structure transformation
  - Context expansion
- Extracts key points more efficiently:
  - Uses improved keyword extraction algorithms
  - Implements better sentence scoring mechanisms
  - Focuses on core spiritual teachings
- Generates more focused training samples:
  - Creates targeted questions based on article content
  - Provides structured answers with article metadata
  - Balances question types for better model learning

### 3. Data Format
The processed data format is as follows:
```json
{
  "question": "Question content",
  "answer": {
    "文章信息": {
      "标题": "Article Title",
      "作者": "Author",
      "卷期": "Volume Number",
      "类别": "Category"
    },
    "主要内容": {
      "概述": "Article overview",
      "关键段落": ["Key paragraph 1", "Key paragraph 2", ...]
    },
    "关键词解释": {
      "keyword1": "Explanation of keyword1",
      "keyword2": "Explanation of keyword2"
    },
    "关键句子解释": {
      "sentence1": "Explanation of sentence1",
      "sentence2": "Explanation of sentence2"
    }
  }
}
```

## Model Training

### 1. Environment Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure GPU environment:
   - CUDA 11.8+
   - PyTorch 2.0+
   - At least 12GB VRAM

### 2. Training Scripts Comparison

#### Standard Training (train.py)
```python
# LoRA Configuration
lora_config = LoraConfig(
    r=32,                # LoRA rank
    lora_alpha=64,       # LoRA alpha value
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Parameters
training_args = TrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    eval_steps=200,
    save_steps=200,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=50
)
```

#### Optimized Training (train_small.py)
```python
# LoRA Configuration
lora_config = LoraConfig(
    r=16,                # Reduced rank for efficiency
    lora_alpha=32,       # Adjusted alpha for balance
    target_modules=[
        "q_proj",
        "v_proj",        # Reduced target modules
    ],
    lora_dropout=0.1,    # Increased dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Parameters
training_args = TrainingArguments(
    num_train_epochs=1,  # Reduced epochs
    per_device_train_batch_size=8,  # Increased batch size
    gradient_accumulation_steps=8,
    eval_steps=50,       # More frequent evaluation
    save_steps=50,       # More frequent saving
    learning_rate=5e-4,  # Adjusted learning rate
    warmup_steps=25,     # Adjusted warmup
    logging_steps=10     # More frequent logging
)
```

### 3. Key Optimizations in train_small.py

1. **Transfer Learning Improvements**:
   - Freezes more layers to reduce trainable parameters
   - Only unfreezes the last few layers (23-27) for fine-tuning
   - Reduces trainable parameters from 2.08% to 0.12%

2. **LoRA Configuration Optimization**:
   - Reduces rank from 32 to 16
   - Decreases target modules from 7 to 2
   - Adjusts alpha from 64 to 32
   - Increases dropout from 0.05 to 0.1

3. **Training Process Optimization**:
   - Reduces training epochs from 2 to 1
   - Increases batch size from 4 to 8
   - More frequent evaluation and saving (every 50 steps)
   - Adds early stopping with patience of 3

4. **Memory Efficiency**:
   - Enables gradient checkpointing
   - Uses mixed precision training (FP16)
   - Optimizes data loading with parallel processing

5. **Training Time Reduction**:
   - Estimated training time reduced from 9-11 hours to ~4 hours
   - Each step takes approximately 44 seconds

### 4. Start Training
```bash
# Standard training
python train.py

# Optimized training
python train_small.py
```

## Model Usage

### 1. Local Deployment
1. Load model and LoRA weights:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from peft import PeftModel
   
   # Load base model
   model = AutoModelForCausalLM.from_pretrained(
       "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
       trust_remote_code=True,
       device_map="auto",
       torch_dtype=torch.float16
   )
   
   # Load LoRA weights
   model = PeftModel.from_pretrained(model, "./results_small")
   tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
   ```

2. Generate responses:
   ```python
   def generate_response(prompt, model, tokenizer):
       inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
       outputs = model.generate(
           **inputs,
           max_length=512,
           temperature=0.6,
           top_p=0.85,
           repetition_penalty=1.3,
           num_beams=3,
           length_penalty=0.8,
           no_repeat_ngram_size=3
       )
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)
       return response
   ```

### 2. Usage Example
```python
# Test questions
test_questions = [
    "What articles has Yang Lei published in ByTheStream magazine?",
    "What articles has Pastor Huang Zhiqi published in ByTheStream magazine?",
    "What is the history of ByTheStream magazine's founding?"
]

# Generate responses
for question in test_questions:
    prompt = f"<think>Please answer the question based on ByTheStream magazine content. Keep the answer concise and cite the source.</think>\n\nQuestion: {question}\n\nAnswer:"
    response = generate_response(prompt, model, tokenizer)
    print(f"Question: {question}")
    print(f"Answer: {response}\n")
```

## Performance Comparison

| Metric | Standard Training | Optimized Training |
|--------|------------------|-------------------|
| Training Time | ~9-11 hours | ~4 hours |
| Memory Usage | ~12GB | ~8GB |
| Trainable Parameters | 2.08% | 0.12% |
| Batch Size | 4 | 8 |
| Steps per Epoch | 337 | 337 |
| Time per Step | ~90 seconds | ~44 seconds |

## Notes
1. The optimized training script (train_small.py) is designed for faster training with minimal quality loss
2. Early stopping mechanism prevents overfitting
3. Checkpoints are saved every 50 steps for better recovery options
4. Training progress is monitored through detailed logging

## Future Plans
1. Further optimize data preparation for even faster training
2. Experiment with different LoRA configurations
3. Implement knowledge distillation for model compression
4. Develop evaluation metrics specific to ByTheStream content 