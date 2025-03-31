import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

class ColabModelSetup:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-base"):
        self.model_name = model_name
        self.model_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'models',
            'deepseek-coder-1.3b-base'
        )
        
    def check_gpu(self):
        """Check if GPU is available and print info."""
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        else:
            print("No GPU available, using CPU")
        
    def download_model(self):
        """Download the model from Hugging Face."""
        print(f"Downloading model {self.model_name}...")
        try:
            snapshot_download(
                repo_id=self.model_name,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", "*.txt"]
            )
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

    def load_model(self):
        """Load the model and tokenizer with Colab optimizations."""
        print("Loading model and tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                low_cpu_mem_usage=True
            )
            
            # Load model with Colab optimizations
            model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            print("Model loaded successfully!")
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def setup(self):
        """Set up the model environment in Colab."""
        self.check_gpu()
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            self.download_model()
        return self.load_model()

if __name__ == "__main__":
    setup = ColabModelSetup()
    model, tokenizer = setup.setup() 