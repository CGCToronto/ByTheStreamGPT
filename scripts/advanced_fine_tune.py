import os
import json
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    getpeft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import logging
from typing import Dict, List, Tuple
import wandb
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import evaluate
from tqdm import tqdm

class AdvancedModelFineTuner:
    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-coder-1.3b-base",
        use_qlora: bool = True,
        use_lora: bool = False,
        n_folds: int = 5
    ):
        self.model_name = model_name
        self.use_qlora = use_qlora
        self.use_lora = use_lora
        self.n_folds = n_folds
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.metrics = {}
        
    def prepare_model(self):
        """Prepare the model and tokenizer with quantization and LoRA if specified."""
        try:
            self.logger.info(f"Loading model and tokenizer from {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Configure quantization if using QLoRA
            if self.use_qlora:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    device_map="auto"
                )
                self.model = prepare_model_for_kbit_training(self.model)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            # Apply LoRA if specified
            if self.use_lora:
                lora_config = LoraConfig(
                    r=16,  # rank
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                self.model = getpeft_model(self.model, lora_config)
            
            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id
                
            self.logger.info("Model and tokenizer loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error preparing model: {e}")
            return False
            
    def prepare_dataset(self, data_path: str):
        """Prepare the dataset for fine-tuning with cross-validation splits."""
        try:
            self.logger.info(f"Loading dataset from {data_path}")
            
            # Load dataset
            self.dataset = load_dataset(
                'json',
                data_files=data_path,
                split='train'
            )
            
            # Tokenize dataset
            def tokenize_function(examples):
                texts = [
                    f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"
                    for example in examples
                ]
                
                tokenized = self.tokenizer(
                    texts,
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                return tokenized
                
            # Apply tokenization
            tokenized_dataset = self.dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=self.dataset.column_names
            )
            
            self.dataset = tokenized_dataset
            self.logger.info("Dataset prepared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {e}")
            return False
            
    def compute_metrics(self, eval_preds):
        """Compute detailed evaluation metrics."""
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=-1)
        
        # Calculate basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Calculate perplexity
        loss = torch.nn.CrossEntropyLoss()(
            torch.tensor(predictions), torch.tensor(labels)
        ).item()
        
        # Calculate BLEU score for text generation
        bleu = evaluate.load("bleu")
        bleu_score = bleu.compute(
            predictions=[self.tokenizer.decode(pred) for pred in predictions],
            references=[[self.tokenizer.decode(label)] for label in labels]
        )["bleu"]
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "loss": loss,
            "bleu": bleu_score
        }
            
    def train_with_cross_validation(self, output_dir: str):
        """Train the model with k-fold cross-validation."""
        try:
            self.logger.info("Starting cross-validation training")
            
            # Initialize k-fold cross-validation
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            
            # Store results for each fold
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
                self.logger.info(f"Training fold {fold + 1}/{self.n_folds}")
                
                # Split dataset
                train_dataset = self.dataset.select(train_idx)
                val_dataset = self.dataset.select(val_idx)
                
                # Training arguments
                training_args = TrainingArguments(
                    output_dir=os.path.join(output_dir, f"fold_{fold + 1}"),
                    num_train_epochs=3,
                    per_device_train_batch_size=4,
                    per_device_eval_batch_size=4,
                    gradient_accumulation_steps=4,
                    learning_rate=2e-5,
                    weight_decay=0.01,
                    warmup_steps=100,
                    logging_steps=10,
                    save_steps=100,
                    eval_steps=100,
                    evaluation_strategy="steps",
                    load_best_model_at_end=True,
                    report_to="wandb",
                    fp16=True,
                    gradient_checkpointing=True,
                    metric_for_best_model="f1"
                )
                
                # Initialize trainer
                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=self.compute_metrics,
                    data_collator=DataCollatorForLanguageModeling(
                        tokenizer=self.tokenizer,
                        mlm=False
                    )
                )
                
                # Train fold
                trainer.train()
                
                # Evaluate fold
                fold_metrics = trainer.evaluate()
                fold_results.append(fold_metrics)
                
                # Save fold model
                trainer.save_model()
                self.tokenizer.save_pretrained(
                    os.path.join(output_dir, f"fold_{fold + 1}")
                )
                
                # Log fold results
                wandb.log({
                    f"fold_{fold + 1}/accuracy": fold_metrics["eval_accuracy"],
                    f"fold_{fold + 1}/f1": fold_metrics["eval_f1"],
                    f"fold_{fold + 1}/bleu": fold_metrics["eval_bleu"]
                })
            
            # Calculate and log average metrics
            avg_metrics = {
                metric: np.mean([fold[metric] for fold in fold_results])
                for metric in fold_results[0].keys()
            }
            
            wandb.log({
                "average/accuracy": avg_metrics["eval_accuracy"],
                "average/f1": avg_metrics["eval_f1"],
                "average/bleu": avg_metrics["eval_bleu"]
            })
            
            self.metrics = avg_metrics
            self.logger.info("Cross-validation training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during cross-validation training: {e}")
            return False
            
    def evaluate(self, test_data_path: str):
        """Evaluate the fine-tuned model with detailed metrics."""
        try:
            self.logger.info("Starting evaluation")
            
            # Load test dataset
            test_dataset = load_dataset(
                'json',
                data_files=test_data_path,
                split='train'
            )
            
            # Prepare test data
            def prepare_test_data(examples):
                texts = [
                    f"Instruction: {example['instruction']}\nInput: {example['input']}"
                    for example in examples
                ]
                return self.tokenizer(
                    texts,
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
            test_dataset = test_dataset.map(
                prepare_test_data,
                batched=True,
                remove_columns=test_dataset.column_names
            )
            
            # Generate responses
            results = []
            for example in tqdm(test_dataset, desc="Generating responses"):
                inputs = self.tokenizer(
                    example['input_ids'],
                    return_tensors='pt'
                ).to(self.model.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append({
                    'input': example['input'],
                    'generated': response
                })
                
            # Calculate additional metrics
            rouge = evaluate.load("rouge")
            bertscore = evaluate.load("bertscore")
            
            rouge_scores = rouge.compute(
                predictions=[r['generated'] for r in results],
                references=[r['input'] for r in results]
            )
            
            bert_scores = bertscore.compute(
                predictions=[r['generated'] for r in results],
                references=[r['input'] for r in results],
                lang="en"
            )
            
            # Save results with metrics
            output_file = os.path.join(
                os.path.dirname(output_dir),
                'evaluation_results.json'
            )
            
            evaluation_results = {
                'responses': results,
                'metrics': {
                    'rouge': rouge_scores,
                    'bertscore': {
                        'mean': float(np.mean(bert_scores['scores'])),
                        'std': float(np.std(bert_scores['scores']))
                    }
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
                
            self.logger.info("Evaluation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize wandb
    wandb.init(project="bythestream-gpt")
    
    # Initialize fine-tuner with QLoRA
    fine_tuner = AdvancedModelFineTuner(
        use_qlora=True,
        use_lora=False,
        n_folds=5
    )
    
    # Prepare model
    if not fine_tuner.prepare_model():
        exit(1)
        
    # Prepare dataset
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'fine_tuning',
        'training_data.jsonl'
    )
    if not fine_tuner.prepare_dataset(data_path):
        exit(1)
        
    # Fine-tune model with cross-validation
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models',
        'fine_tuned'
    )
    if not fine_tuner.train_with_cross_validation(output_dir):
        exit(1)
        
    # Evaluate model
    test_data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'fine_tuning',
        'test_data.jsonl'
    )
    if not fine_tuner.evaluate(test_data_path):
        exit(1)
        
    # Close wandb
    wandb.finish() 