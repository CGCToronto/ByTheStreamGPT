import logging
from deepseek_handler import DeepSeekHandler
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    """Test the DeepSeek model with various queries."""
    try:
        # Initialize handler
        handler = DeepSeekHandler()
        logger.info("Checking GPU availability...")
        handler.check_gpu()
        
        # Set up model
        logger.info("Setting up model...")
        start_time = time.time()
        handler.setup()
        setup_time = time.time() - start_time
        logger.info(f"Model setup completed in {setup_time:.2f} seconds")
        
        # Test queries
        test_queries = [
            "What does the Bible say about love?",
            "How can I improve my prayer life?",
            "What is the meaning of faith?",
            "How to study the Bible effectively?"
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting query: {query}")
            start_time = time.time()
            response = handler.generate_response(query)
            response_time = time.time() - start_time
            
            print(f"\nQuery: {query}")
            print(f"Response: {response}")
            print(f"Response time: {response_time:.2f} seconds")
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    test_model() 