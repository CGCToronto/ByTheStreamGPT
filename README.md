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
1. Run the data preprocessing script:
   ```bash
   python prepare_data.py
   ```
2. The script will:
   - Clean HTML tags
   - Standardize text format
   - Generate Q&A pairs for training
   - Create training and validation sets

### 3. Data Format
The processed data format is as follows:
```json
{
  "instruction": "Please answer the question based on ByTheStream magazine content. Keep the answer concise and cite the source.",
  "input": "Question content",
  "output": "Based on [Volume X, 'Title', Author], the answer is...",
  "history": []
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

### 2. Training Configuration
Current LoRA configuration:
```python
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
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 3. Start Training
```bash
python train.py
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
   model = PeftModel.from_pretrained(model, "./fine_tuned_model")
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

## Notes
1. Ensure sufficient GPU memory (recommended at least 12GB)
2. Checkpoints are saved periodically during training
3. Early stopping mechanism is used to prevent overfitting
4. Best model is automatically saved after training

## Current Status
- Training in progress
- Using early stopping to monitor training progress
- Saving checkpoints every 50 steps
- Using TensorBoard to record training process

## Future Plans
1. Evaluate model performance
2. Optimize generation parameters
3. Conduct more test case validation
4. Adjust model based on feedback 