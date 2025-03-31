import os
import logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from config.config import (
    MAX_LENGTH,
    TEMPERATURE,
    TOP_P,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)

class LocalModelHandler:
    def __init__(self, model_path: str):
        """Initialize the local model handler."""
        logger.info(f"Loading model from {model_path}")
        
        # Force CUDA usage
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your GPU installation.")
            
        self.device = "cuda"
        logger.info(f"Using device: {self.device}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # Always use half precision for GPU
                device_map="auto",
                low_cpu_mem_usage=True
            )
            # Move model to GPU explicitly
            self.model = self.model.to(self.device)
            # Enable model evaluation mode
            self.model.eval()
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def process_query(
        self,
        text: str,
        max_length: Optional[int] = None,
        language: str = DEFAULT_LANGUAGE,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P
    ) -> str:
        """Process a query and return the response."""
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}. Must be one of {SUPPORTED_LANGUAGES}")
            
        # Prepare the prompt
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {text}\n\nAssistant:"
        
        # Set generation parameters
        max_length = max_length or MAX_LENGTH
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            # Decode and return response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
            
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "model_type": self.model.config.model_type,
            "model_size": sum(p.numel() for p in self.model.parameters()) / 1e6,  # in millions
            "tokenizer_vocab_size": len(self.tokenizer),
            "supports_half_precision": self.device == "cuda"
        }

if __name__ == "__main__":
    # Example usage
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models',
        'deepseek-coder-1.3b-base'
    )
    
    handler = LocalModelHandler(model_path)
    
    # Test query
    result = handler.process_query("What does the Bible say about love?")
    print(result) 