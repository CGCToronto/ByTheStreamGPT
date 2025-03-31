import os
import sys
import logging
import signal
from datetime import datetime
from pipeline_manager import PipelineManager

def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'logs'
    )
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(
        log_dir,
        f'pipeline_{datetime.now().strftime("%Y%m%d")}.log'
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main function to run the pipeline service."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Get website directory
        website_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'bythestreamwebsite'
        )
        
        if not os.path.exists(website_dir):
            logger.error(f"Website directory not found: {website_dir}")
            sys.exit(1)
            
        # Initialize pipeline manager
        pipeline = PipelineManager(website_dir)
        
        # Handle shutdown signals
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            pipeline.stop_monitoring()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Process existing volumes
        logger.info("Processing existing volumes...")
        pipeline.process_all_volumes()
        
        # Start monitoring for new volumes
        logger.info("Starting volume monitoring...")
        pipeline.start_monitoring()
        
        # Keep the service running
        while True:
            signal.pause()
            
    except Exception as e:
        logger.error(f"Error in pipeline service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 