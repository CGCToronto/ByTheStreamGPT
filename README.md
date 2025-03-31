# ByTheStreamGPT

A specialized GPT model fine-tuned on Christian magazine content from By The Stream.

## Local Development

1. Set up the environment:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up local test environment
python scripts/deploy.py --setup
```

2. Run local tests:
```bash
python scripts/deploy.py --test
```

The API will be available at `http://localhost:7860`

## Deployment to Hugging Face Spaces

1. Prepare deployment files:
```bash
python scripts/deploy.py --prepare
```

2. Push to Hugging Face:
```bash
# Login to Hugging Face (if not already logged in)
huggingface-cli login

# Create a new Space (if not already created)
huggingface-cli space create bythestream-gpt

# Push the code
cd deploy
git init
git add .
git commit -m "Initial deployment"
git remote add origin https://huggingface.co/spaces/your-username/bythestream-gpt
git push -u origin main
```

3. Monitor deployment:
- Visit your Space URL
- Check the deployment logs in the Hugging Face dashboard
- Test the API endpoints

## API Endpoints

- `GET /`: Health check
- `POST /query`: Process a query
  ```json
  {
    "text": "你的问题",
    "language": "simplified",
    "max_length": 200
  }
  ```
- `POST /update`: Update model with new volumes
  ```json
  {
    "force_update": false
  }
  ```
- `GET /info`: Get model information

## Model Updates

The model can be updated with new volumes through the `/update` endpoint. This will:
1. Process new volumes
2. Update the vector database
3. Fine-tune the model
4. Reload with new weights

## Directory Structure

```
ByTheStreamGPT/
├── app.py                 # FastAPI application
├── config/               # Configuration files
├── data/                 # Data directory
│   ├── models/          # Model files
│   ├── vector_db/       # Vector database
│   └── test/            # Test data
├── scripts/             # Utility scripts
├── deploy/              # Deployment files
└── requirements.txt     # Python dependencies
```

## Model Details

- Name: DeepSeek-Coder-1.3B
- Size: ~1.3GB
- Memory Usage: ~2.6GB
- GPU Memory: ~1.3GB (FP16)

## Performance

- Average response time: < 2 seconds
- GPU acceleration when available
- Memory-optimized loading
- Efficient inference

## License

MIT License 