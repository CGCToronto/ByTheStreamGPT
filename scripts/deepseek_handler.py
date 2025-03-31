import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import os
from typing import Dict, List, Optional
import logging
from .content_manager import ContentManager
from .spiritual_knowledge import SpiritualKnowledge
import gc
import re

class DeepSeekHandler:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-1.3b-base"):
        self.model_name = model_name
        self.model_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'models',
            'deepseek-coder-1.3b-base'
        )
        self.model = None
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)
        self.content_manager = ContentManager(os.path.dirname(os.path.dirname(__file__)))
        self.spiritual_knowledge = SpiritualKnowledge()
        self.max_context_length = 1000  # Maximum context length for model input
        
    def check_gpu(self) -> bool:
        """Check if GPU is available and print info."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"GPU available: {gpu_name}")
            self.logger.info(f"GPU memory: {gpu_memory:.2f}GB")
            return True
        self.logger.info("No GPU available, using CPU")
        return False
        
    def download_model(self):
        """Download the model from Hugging Face."""
        self.logger.info(f"Downloading model {self.model_name}...")
        try:
            snapshot_download(
                repo_id=self.model_name,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", "*.txt"]
            )
            self.logger.info("Model downloaded successfully!")
        except Exception as e:
            self.logger.error(f"Error downloading model: {e}")
            raise

    def load_model(self):
        """Load the model and tokenizer with optimizations."""
        self.logger.info("Loading model and tokenizer...")
        try:
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                low_cpu_mem_usage=True
            )
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.logger.info("Model loaded successfully!")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def _truncate_context(self, context: str) -> str:
        """Truncate context to fit within model's context window."""
        if len(context) > self.max_context_length:
            return context[:self.max_context_length] + "..."
        return context

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract potential spiritual keywords from text."""
        # Common spiritual terms
        spiritual_terms = [
            "love", "faith", "grace", "prayer", "bible", "god", "jesus", "christ",
            "holy spirit", "salvation", "sin", "forgiveness", "mercy", "hope",
            "peace", "joy", "wisdom", "truth", "light", "darkness", "eternal life",
            "heaven", "hell", "gospel", "worship", "blessing", "curse", "righteousness",
            "justice", "mercy", "compassion", "obedience", "discipleship", "ministry"
        ]
        
        # Convert to lowercase for case-insensitive matching
        text = text.lower()
        keywords = []
        
        # Find matches
        for term in spiritual_terms:
            if term in text:
                keywords.append(term)
                
        return keywords
        
    def _get_spiritual_context(self, prompt: str) -> str:
        """Get spiritual context based on keywords in the prompt."""
        keywords = self._extract_keywords(prompt)
        spiritual_context = ""
        
        for keyword in keywords:
            # Get concept information
            concept_info = self.spiritual_knowledge.get_concept_info(keyword)
            if concept_info:
                spiritual_context += self.spiritual_knowledge.format_concept_prompt(keyword)
                
            # Get related concepts
            related = self.spiritual_knowledge.get_related_concepts(keyword)
            for related_concept in related:
                related_info = self.spiritual_knowledge.get_concept_info(related_concept)
                if related_info:
                    spiritual_context += self.spiritual_knowledge.format_concept_prompt(related_concept)
                    
        return spiritual_context

    def generate_response(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        volume: Optional[str] = None
    ) -> Dict:
        """Generate a response from the model with relevant articles and spiritual knowledge."""
        try:
            # Get relevant articles
            relevant_articles = self.content_manager.get_relevant_articles(prompt, volume)
            
            # Get spiritual context
            spiritual_context = self._get_spiritual_context(prompt)
            
            # Format context with relevant articles and spiritual knowledge
            context = "Based on the following information:\n\n"
            
            # Add spiritual context if available
            if spiritual_context:
                context += "Spiritual Knowledge:\n"
                context += spiritual_context
                context += "\n"
            
            # Add relevant articles
            if relevant_articles:
                context += "Relevant Articles:\n"
                for article in relevant_articles:
                    context += f"From Volume {article.get('volume')}:\n{article.get('content')}\n\n"
            
            # Truncate context if needed
            context = self._truncate_context(context)
            
            # Combine context with prompt
            full_prompt = f"{context}\nQuestion: {prompt}\nAnswer:"
            
            # Generate response
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clear memory
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return {
                "success": True,
                "response": response,
                "relevant_articles": [
                    {
                        "volume": article.get('volume'),
                        "title": article.get('title'),
                        "id": article.get('id')
                    }
                    for article in relevant_articles
                ],
                "spiritual_keywords": self._extract_keywords(prompt)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def setup(self):
        """Set up the model environment."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            self.download_model()
        return self.load_model()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize and test the model
    handler = DeepSeekHandler()
    handler.check_gpu()
    handler.setup()
    
    # Test query
    test_query = "What does the Bible say about love?"
    result = handler.generate_response(test_query)
    print(f"Query: {test_query}")
    print(f"Response: {result['response']}")
    print("\nRelevant articles:")
    for article in result['relevant_articles']:
        print(f"- {article['title']} (Volume {article['volume']})") 