import os
import json
import logging
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from config.config import (
    DATA_DIR,
    MODELS_DIR,
    MODEL_NAME,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MAX_LENGTH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.data_dir = Path(DATA_DIR)
        self.models_dir = Path(MODELS_DIR)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        logger.info(f"Loading model {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Add special tokens if needed
        special_tokens = {
            "pad_token": "[PAD]",
            "sep_token": "[SEP]",
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def load_data(self):
        """Load and prepare the training data."""
        data_file = self.data_dir / 'processed_articles.json'
        with open(data_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        # Prepare training examples
        training_examples = []
        for article in articles:
            # Format: theme, content pair for both simplified and traditional
            example = {
                'text': f"主题: {article['theme']}\n内容: {article['content']}\n[SEP]"
            }
            training_examples.append(example)
        
        return Dataset.from_list(training_examples)

    @staticmethod
    def preprocess_function(examples, tokenizer, max_length):
        """Tokenize the text data."""
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )

    def train(self):
        """Train the model on the processed data."""
        # Load and preprocess data
        dataset = self.load_data()
        tokenized_dataset = dataset.map(
            lambda x: self.preprocess_function(x, self.tokenizer, MAX_LENGTH),
            remove_columns=dataset.column_names,
            num_proc=4
        )

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(self.models_dir / "checkpoints"),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            fp16=False,
            bf16=torch.cuda.is_available(),
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            optim="adamw_torch"
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )

        # Train the model
        logger.info("Starting training...")
        trainer.train()

        # Save the final model
        output_dir = self.models_dir / "latest"
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train() 