import firebase_functions as functions
import firebase_admin
from firebase_admin import credentials, firestore
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# Initialize Firebase Admin
cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred)
db = firestore.client()

class FirebaseModelHandler:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-base"):
        self.model_name = model_name
        self.model_dir = "/tmp/models/deepseek-coder-1.3b-base"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer with Firebase optimizations."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            snapshot_download(
                repo_id=self.model_name,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", "*.txt"]
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            low_cpu_mem_usage=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

# Initialize model handler
model_handler = FirebaseModelHandler()

@functions.https.on_request()
def process_query(request):
    """Process a user query and return response."""
    try:
        # Load model if not loaded
        if model_handler.model is None:
            model_handler.load_model()
            
        # Get request data
        request_json = request.get_json()
        query = request_json.get('query')
        language = request_json.get('language')
        
        if not query:
            return functions.Response(
                'No query provided',
                status=400
            )
            
        # Process query
        inputs = model_handler.tokenizer(query, return_tensors="pt")
        outputs = model_handler.model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        response = model_handler.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return functions.Response(
            response,
            status=200
        )
        
    except Exception as e:
        return functions.Response(
            f'Error processing query: {str(e)}',
            status=500
        )

@functions.https.on_request()
def health_check(request):
    """Health check endpoint."""
    return functions.Response(
        'Service is healthy',
        status=200
    ) 