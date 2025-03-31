import os
import logging
import subprocess
from pathlib import Path
import shutil
import json
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import BASE_DIR, DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_local_test():
    """Set up the environment for local testing."""
    logger.info("Setting up local test environment...")
    
    # Create necessary directories
    for dir_path in [DATA_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        
    # Create test data if needed
    test_data_dir = DATA_DIR / 'test'
    test_data_dir.mkdir(exist_ok=True)
    
    # Create a test volume
    test_volume = test_data_dir / 'volume_1'
    test_volume.mkdir(exist_ok=True)
    
    # Create test metadata
    metadata = {
        'theme_simplified': '测试主题',
        'theme_traditional': '測試主題',
        'theme_english': 'Test Theme',
        'year': '2024',
        'month': '1'
    }
    
    with open(test_volume / 'metadata.txt', 'w', encoding='utf-8') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
            
    logger.info("Local test environment setup complete.")

def test_local():
    """Run local tests."""
    logger.info("Running local tests...")
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Run the FastAPI server
    try:
        subprocess.run(['uvicorn', 'app:app', '--reload', '--host', '0.0.0.0', '--port', '7860'], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running local server: {e}")
        return False
        
    return True

def prepare_huggingface_deployment():
    """Prepare files for Hugging Face deployment."""
    logger.info("Preparing files for Hugging Face deployment...")
    
    # Create deployment directory
    deploy_dir = BASE_DIR / 'deploy'
    deploy_dir.mkdir(exist_ok=True)
    
    # Copy necessary files
    files_to_copy = [
        'app.py',
        'requirements.txt',
        'Dockerfile',
        '.huggingface/config.yaml',
        'README.md'
    ]
    
    for file in files_to_copy:
        src = BASE_DIR / file
        if src.exists():
            shutil.copy2(src, deploy_dir)
            
    # Create deployment-specific requirements
    deploy_requirements = [
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'peft>=0.4.0',
        'accelerate>=0.20.0',
        'bitsandbytes>=0.41.0',
        'fastapi>=0.100.0',
        'uvicorn>=0.22.0',
        'pydantic>=2.0.0',
        'sentencepiece>=0.1.99',
        'protobuf>=4.23.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.2.0',
        'tqdm>=4.65.0',
        'wandb>=0.15.0',
        'faiss-cpu>=1.7.4',
        'sentence-transformers>=2.2.0'
    ]
    
    with open(deploy_dir / 'requirements.txt', 'w') as f:
        f.write('\n'.join(deploy_requirements))
        
    logger.info("Hugging Face deployment files prepared.")

def main():
    """Main deployment script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deployment script for ByTheStreamGPT')
    parser.add_argument('--setup', action='store_true', help='Set up local test environment')
    parser.add_argument('--test', action='store_true', help='Run local tests')
    parser.add_argument('--prepare', action='store_true', help='Prepare files for Hugging Face deployment')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_local_test()
    if args.test:
        test_local()
    if args.prepare:
        prepare_huggingface_deployment()
        
if __name__ == "__main__":
    main() 