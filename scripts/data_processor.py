import os
import json
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from config.config import VOLUMES_DIR, DATA_DIR

class DataProcessor:
    def __init__(self):
        self.volumes_dir = Path(VOLUMES_DIR)
        self.data_dir = Path(DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def read_metadata(self, volume_dir: Path) -> Dict:
        """Read metadata from a volume directory."""
        metadata_file = volume_dir / "metadata.txt"
        if not metadata_file.exists():
            return {}
        
        metadata = {}
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                key, value = line.strip().split(': ', 1)
                metadata[key] = value
        return metadata

    def read_article(self, file_path: Path) -> Optional[Dict]:
        """Read an article file and return its content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Extract article number and author from filename
            filename = file_path.stem
            parts = filename.split('_')
            if len(parts) >= 2:
                article_number = parts[0]
                author = parts[1]
                language = 'traditional' if filename.endswith('_t') else 'simplified'
            else:
                article_number = filename
                author = "unknown"
                language = 'simplified'

            return {
                'content': content,
                'article_number': article_number,
                'author': author,
                'language': language,
                'file_path': str(file_path)
            }
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def process_volume(self, volume_dir: Path) -> List[Dict]:
        """Process all articles in a volume directory."""
        articles = []
        metadata = self.read_metadata(volume_dir)
        volume_number = volume_dir.name.split('_')[1]

        for file_path in volume_dir.glob('*.txt'):
            if file_path.name == 'metadata.txt':
                continue
            
            article_data = self.read_article(file_path)
            if article_data:
                article_data.update({
                    'volume': volume_number,
                    'theme': metadata.get(f'theme_{article_data["language"]}', ''),
                    'theme_english': metadata.get('theme_english', ''),
                    'year': metadata.get('year', ''),
                    'month': metadata.get('month', '')
                })
                articles.append(article_data)
        
        return articles

    def process_all_volumes(self) -> List[Dict]:
        """Process all volumes and their articles."""
        all_articles = []
        volume_dirs = sorted(self.volumes_dir.glob('volume_*'))

        for volume_dir in tqdm(volume_dirs, desc="Processing volumes"):
            articles = self.process_volume(volume_dir)
            all_articles.extend(articles)

        return all_articles

    def save_processed_data(self, articles: List[Dict]):
        """Save processed articles to JSON file."""
        output_file = self.data_dir / 'processed_articles.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

    def create_article_index(self, articles: List[Dict]) -> pd.DataFrame:
        """Create a searchable index of articles."""
        df = pd.DataFrame(articles)
        index_file = self.data_dir / 'article_index.csv'
        df.to_csv(index_file, index=False, encoding='utf-8')
        return df

    def process(self):
        """Main processing function."""
        print("Starting data processing...")
        articles = self.process_all_volumes()
        print(f"Processed {len(articles)} articles")
        
        self.save_processed_data(articles)
        df = self.create_article_index(articles)
        print(f"Created index with {len(df)} entries")
        
        return articles, df

if __name__ == "__main__":
    processor = DataProcessor()
    articles, df = processor.process() 