import os
import logging
import json
from datetime import datetime
from typing import Dict, List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .content_manager import ContentManager
from .deepseek_handler import DeepSeekHandler

class VolumeUpdateHandler(FileSystemEventHandler):
    def __init__(self, pipeline_manager):
        self.pipeline_manager = pipeline_manager
        
    def on_created(self, event):
        if event.is_directory and event.src_path.startswith(self.pipeline_manager.website_dir):
            # Check if it's a volume directory
            if "volume_" in event.src_path:
                volume = event.src_path.split("volume_")[-1]
                self.pipeline_manager.logger.info(f"New volume detected: {volume}")
                self.pipeline_manager.process_new_volume(volume)

class PipelineManager:
    def __init__(self, website_dir: str):
        self.website_dir = website_dir
        self.logger = logging.getLogger(__name__)
        self.content_manager = ContentManager(website_dir)
        self.model_handler = DeepSeekHandler()
        self.observer = None
        self.last_processed_volumes = set()
        
    def start_monitoring(self):
        """Start monitoring for new volumes."""
        self.logger.info("Starting volume monitoring...")
        self.observer = Observer()
        event_handler = VolumeUpdateHandler(self)
        self.observer.schedule(event_handler, self.website_dir, recursive=False)
        self.observer.start()
        self.logger.info("Volume monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring for new volumes."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.logger.info("Volume monitoring stopped")
            
    def process_new_volume(self, volume: str):
        """Process a newly added volume."""
        try:
            self.logger.info(f"Processing new volume: {volume}")
            
            # Check if volume was already processed
            if volume in self.last_processed_volumes:
                self.logger.info(f"Volume {volume} already processed")
                return
                
            # Update content cache
            self.content_manager.update_cache()
            
            # Generate volume summary
            summary = self._generate_volume_summary(volume)
            
            # Save summary
            self._save_volume_summary(volume, summary)
            
            # Update processed volumes
            self.last_processed_volumes.add(volume)
            
            self.logger.info(f"Successfully processed volume {volume}")
            
        except Exception as e:
            self.logger.error(f"Error processing volume {volume}: {e}")
            
    def _generate_volume_summary(self, volume: str) -> Dict:
        """Generate a summary of the volume's content."""
        try:
            # Get articles from the volume
            articles = self.content_manager.load_articles(volume)
            
            # Generate summary using the model
            prompt = f"Please provide a brief summary of the main themes and key messages from Volume {volume} of the Christian magazine."
            result = self.model_handler.generate_response(prompt, volume=volume)
            
            return {
                "volume": volume,
                "summary": result.get("response", ""),
                "article_count": len(articles),
                "processed_date": datetime.now().isoformat(),
                "articles": [
                    {
                        "id": article.get("id"),
                        "title": article.get("title")
                    }
                    for article in articles
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating volume summary: {e}")
            return {
                "volume": volume,
                "error": str(e),
                "processed_date": datetime.now().isoformat()
            }
            
    def _save_volume_summary(self, volume: str, summary: Dict):
        """Save the volume summary to a file."""
        try:
            summaries_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'data',
                'summaries'
            )
            os.makedirs(summaries_dir, exist_ok=True)
            
            summary_file = os.path.join(summaries_dir, f"volume_{volume}_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Saved summary for volume {volume}")
            
        except Exception as e:
            self.logger.error(f"Error saving volume summary: {e}")
            
    def process_all_volumes(self):
        """Process all existing volumes."""
        try:
            volumes = [d for d in os.listdir(self.website_dir) if d.startswith('volume_')]
            for volume in volumes:
                volume_num = volume.split('_')[1]
                self.process_new_volume(volume_num)
                
        except Exception as e:
            self.logger.error(f"Error processing all volumes: {e}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize pipeline manager
    website_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'bythestreamwebsite'
    )
    pipeline = PipelineManager(website_dir)
    
    # Process existing volumes
    pipeline.process_all_volumes()
    
    # Start monitoring for new volumes
    pipeline.start_monitoring()
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pipeline.stop_monitoring() 