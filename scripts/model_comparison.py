import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import evaluate
from datetime import datetime
import wandb

class ModelComparator:
    def __init__(self, base_model_name: str = "deepseek-ai/deepseek-coder-1.3b-base"):
        self.base_model_name = base_model_name
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.tokenizer = None
        self.metrics = {}
        
    def load_model(self, model_path: str, model_name: str):
        """Load a model from the specified path."""
        try:
            self.logger.info(f"Loading model from {model_path}")
            
            # Load tokenizer if not already loaded
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            self.models[model_name] = model
            self.logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            return False
            
    def generate_response(
        self,
        model: AutoModelForCausalLM,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate a response from the model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def evaluate_model(
        self,
        model: AutoModelForCausalLM,
        test_data: List[Dict],
        model_name: str
    ) -> Dict:
        """Evaluate a model on test data."""
        try:
            self.logger.info(f"Evaluating model {model_name}")
            
            results = []
            for example in tqdm(test_data, desc=f"Generating responses for {model_name}"):
                prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}"
                response = self.generate_response(model, prompt)
                
                results.append({
                    'input': example['input'],
                    'generated': response,
                    'reference': example['output']
                })
            
            # Calculate metrics
            bleu = evaluate.load("bleu")
            rouge = evaluate.load("rouge")
            bertscore = evaluate.load("bertscore")
            
            bleu_scores = bleu.compute(
                predictions=[r['generated'] for r in results],
                references=[[r['reference']] for r in results]
            )
            
            rouge_scores = rouge.compute(
                predictions=[r['generated'] for r in results],
                references=[r['reference'] for r in results]
            )
            
            bert_scores = bertscore.compute(
                predictions=[r['generated'] for r in results],
                references=[r['reference'] for r in results],
                lang="en"
            )
            
            # Calculate response time
            response_times = []
            for example in tqdm(test_data[:10], desc=f"Measuring response time for {model_name}"):
                prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}"
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                self.generate_response(model, prompt)
                end_time.record()
                
                torch.cuda.synchronize()
                response_times.append(start_time.elapsed_time(end_time))
            
            avg_response_time = np.mean(response_times)
            
            metrics = {
                'bleu': bleu_scores['bleu'],
                'rouge': rouge_scores,
                'bertscore': {
                    'mean': float(np.mean(bert_scores['scores'])),
                    'std': float(np.std(bert_scores['scores']))
                },
                'avg_response_time': avg_response_time
            }
            
            self.metrics[model_name] = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_name}: {e}")
            return None
            
    def compare_models(self, test_data_path: str):
        """Compare all loaded models."""
        try:
            # Load test data
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = [json.loads(line) for line in f]
            
            # Evaluate each model
            for model_name, model in self.models.items():
                metrics = self.evaluate_model(model, test_data, model_name)
                if metrics:
                    self.logger.info(f"Metrics for {model_name}:")
                    self.logger.info(json.dumps(metrics, indent=2))
                    
                    # Log to wandb
                    wandb.log({
                        f"{model_name}/bleu": metrics['bleu'],
                        f"{model_name}/rouge1": metrics['rouge']['rouge1'],
                        f"{model_name}/rouge2": metrics['rouge']['rouge2'],
                        f"{model_name}/rougeL": metrics['rouge']['rougeL'],
                        f"{model_name}/bertscore_mean": metrics['bertscore']['mean'],
                        f"{model_name}/avg_response_time": metrics['avg_response_time']
                    })
            
            # Save comparison results
            output_file = os.path.join(
                os.path.dirname(test_data_path),
                'model_comparison_results.json'
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, ensure_ascii=False, indent=2)
                
            self.logger.info("Model comparison completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during model comparison: {e}")
            return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize wandb
    wandb.init(project="bythestream-gpt-comparison")
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Load models to compare
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models',
        'fine_tuned'
    )
    
    # Load base model
    if not comparator.load_model(
        comparator.base_model_name,
        "base_model"
    ):
        exit(1)
    
    # Load fine-tuned models
    for fold in range(1, 6):
        model_path = os.path.join(models_dir, f"fold_{fold}")
        if not comparator.load_model(
            model_path,
            f"fold_{fold}"
        ):
            exit(1)
    
    # Compare models
    test_data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'fine_tuning',
        'test_data.jsonl'
    )
    if not comparator.compare_models(test_data_path):
        exit(1)
    
    # Close wandb
    wandb.finish() 