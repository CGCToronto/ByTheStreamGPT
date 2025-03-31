import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEBSITE_DIR = BASE_DIR.parent / 'ByTheStreamWebsite'
VOLUMES_DIR = WEBSITE_DIR / 'public' / 'text'
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# Model settings
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
MAX_LENGTH = 512
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

# Language settings
LANGUAGES = ['simplified', 'traditional']

# Training settings
TRAIN_TEST_SPLIT = 0.2
RANDOM_SEED = 42
NUM_FOLDS = 5

# Chrome extension settings
MAX_RESPONSE_LENGTH = 200
TEMPERATURE = 0.7
TOP_P = 0.9

# Logging settings
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create necessary directories
for directory in [DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# Model Configuration
MAX_TOKENS = 2000
TEMPERATURE = 0.7
TOP_P = 0.9
FREQUENCY_PENALTY = 0.5
PRESENCE_PENALTY = 0.5

# Vector Database Configuration
VECTOR_DB_DIR = os.path.join(DATA_DIR, 'vector_db')
COLLECTION_NAME = "bythestream_articles"

# Language Configuration
SUPPORTED_LANGUAGES = ['simplified', 'traditional']
DEFAULT_LANGUAGE = 'simplified'

# System Prompts
SYSTEM_PROMPT = """You are a specialized AI assistant for the "溪水旁" (By The Stream) Christian magazine.
Your role is to:
1. Help users find relevant articles and content
2. Provide spiritual guidance based on the magazine's content
3. Answer questions about biblical topics
4. Assist with both simplified and traditional Chinese content
5. Maintain a respectful and spiritually appropriate tone

Always base your responses on the content from the magazine and biblical principles."""

# Error Messages
ERROR_MESSAGES = {
    'api_error': 'There was an error processing your request. Please try again later.',
    'no_results': 'No relevant articles found for your query.',
    'invalid_language': 'Invalid language selection. Please choose between simplified and traditional Chinese.',
    'content_not_found': 'The requested content could not be found.',
} 