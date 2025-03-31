import os
import logging
from pathlib import Path
import json
import shutil
from datetime import datetime
from typing import List, Dict, Optional

from config.config import (
    VOLUMES_DIR,
    DATA_DIR,
    MODELS_DIR,
    MODEL_NAME
)

logger = logging.getLogger(__name__)

class VolumeUpdater:
    def __init__(self):
        """Initialize the volume updater."""
        self.processed_volumes_file = DATA_DIR / 'processed' / 'volumes.json'
        self.processed_volumes_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load processed volumes
        self.processed_volumes = self._load_processed_volumes()
        
    def _load_processed_volumes(self) -> Dict[str, Dict]:
        """Load the list of processed volumes."""
        if self.processed_volumes_file.exists():
            try:
                with open(self.processed_volumes_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error("Error reading processed volumes file")
                return {}
        return {}
        
    def _save_processed_volumes(self):
        """Save the list of processed volumes."""
        with open(self.processed_volumes_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_volumes, f, ensure_ascii=False, indent=2)
            
    def find_new_volumes(self) -> List[Path]:
        """Find volumes that haven't been processed yet."""
        new_volumes = []
        
        for volume_dir in VOLUMES_DIR.glob('volume_*'):
            volume_id = volume_dir.name
            if volume_id not in self.processed_volumes:
                new_volumes.append(volume_dir)
                
        return new_volumes
        
    def process_new_volumes(self) -> bool:
        """Process new volumes and update the model."""
        try:
            new_volumes = self.find_new_volumes()
            if not new_volumes:
                logger.info("No new volumes to process")
                return True
                
            logger.info(f"Found {len(new_volumes)} new volumes to process")
            
            # Process each new volume
            for volume_dir in new_volumes:
                volume_id = volume_dir.name
                
                # Read metadata
                metadata_file = volume_dir / 'metadata.txt'
                if not metadata_file.exists():
                    logger.warning(f"No metadata found for {volume_id}")
                    continue
                    
                metadata = {}
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        key, value = line.strip().split(':', 1)
                        metadata[key.strip()] = value.strip()
                        
                # Process articles
                articles = []
                for article_file in volume_dir.glob('*.txt'):
                    if article_file.name == 'metadata.txt':
                        continue
                        
                    with open(article_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    article_info = {
                        'filename': article_file.name,
                        'content': content,
                        'processed_date': datetime.now().isoformat()
                    }
                    articles.append(article_info)
                    
                # Save volume info
                self.processed_volumes[volume_id] = {
                    'metadata': metadata,
                    'articles': articles,
                    'processed_date': datetime.now().isoformat()
                }
                
            # Save processed volumes
            self._save_processed_volumes()
            
            # Backup current model if it exists
            latest_model_dir = MODELS_DIR / 'latest'
            if latest_model_dir.exists():
                backup_dir = MODELS_DIR / f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                shutil.copytree(latest_model_dir, backup_dir)
                
            logger.info("Volume processing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error processing volumes: {e}")
            return False
            
    def get_model_info(self) -> Dict:
        """Get information about the current model state."""
        latest_model_dir = MODELS_DIR / 'latest'
        
        info = {
            'base_model': MODEL_NAME,
            'processed_volumes': len(self.processed_volumes),
            'latest_update': None,
            'total_articles': 0
        }
        
        if self.processed_volumes:
            latest_volume = max(
                self.processed_volumes.values(),
                key=lambda x: x['processed_date']
            )
            info['latest_update'] = latest_volume['processed_date']
            
            for volume in self.processed_volumes.values():
                info['total_articles'] += len(volume['articles'])
                
        if latest_model_dir.exists():
            info['model_path'] = str(latest_model_dir)
            
        return info

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run updater
    updater = VolumeUpdater()
    success = updater.process_new_volumes()
    
    if success:
        info = updater.get_model_info()
        print("\nModel Information:")
        print(f"Total Processed Volumes: {info.get('processed_volumes', 0)}")
        print(f"Last Update: {info.get('latest_update', 'Unknown')}")
        print(f"Model Path: {info.get('model_path', 'Unknown')}")
    else:
        print("\nError occurred during update process.") 