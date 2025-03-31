import os
import json
from typing import List, Dict
import logging
from datetime import datetime
from content_manager import ContentManager
from spiritual_knowledge import SpiritualKnowledge

class FineTuningDataPreparator:
    def __init__(self, website_dir: str):
        self.website_dir = website_dir
        self.logger = logging.getLogger(__name__)
        self.content_manager = ContentManager(website_dir)
        self.spiritual_knowledge = SpiritualKnowledge()
        
    def prepare_training_data(self) -> List[Dict]:
        """Prepare training data from magazine content and spiritual knowledge."""
        try:
            training_data = []
            
            # Update content cache
            self.content_manager.update_cache()
            
            # Process each volume
            for volume, articles in self.content_manager.articles_cache.items():
                for article in articles:
                    # Get full article content
                    full_article = self.content_manager.get_article_by_id(article['id'])
                    if not full_article:
                        continue
                        
                    # Extract themes and verses
                    themes = article.get('themes', [])
                    verses = article.get('verses', [])
                    
                    # Create training examples
                    examples = self._create_training_examples(full_article, themes, verses)
                    training_data.extend(examples)
                    
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return []
            
    def _create_training_examples(self, article: Dict, themes: List[str], verses: List[str]) -> List[Dict]:
        """Create training examples from an article."""
        examples = []
        
        # Example 1: Article summary
        examples.append({
            "instruction": "Please provide a brief summary of this article.",
            "input": article.get('content', ''),
            "output": self._generate_summary(article)
        })
        
        # Example 2: Theme analysis
        if themes:
            examples.append({
                "instruction": "What are the main spiritual themes in this article?",
                "input": article.get('content', ''),
                "output": self._analyze_themes(article, themes)
            })
            
        # Example 3: Verse analysis
        if verses:
            examples.append({
                "instruction": "How does this article interpret and apply these Bible verses?",
                "input": f"Article: {article.get('content', '')}\nVerses: {', '.join(verses)}",
                "output": self._analyze_verses(article, verses)
            })
            
        # Example 4: Spiritual application
        examples.append({
            "instruction": "What are the practical spiritual applications from this article?",
            "input": article.get('content', ''),
            "output": self._generate_applications(article)
        })
        
        return examples
        
    def _generate_summary(self, article: Dict) -> str:
        """Generate a summary of the article."""
        return f"This article, titled '{article.get('title', '')}', discusses {article.get('content', '')[:200]}..."
        
    def _analyze_themes(self, article: Dict, themes: List[str]) -> str:
        """Analyze spiritual themes in the article."""
        theme_analysis = f"The article explores several spiritual themes:\n"
        for theme in themes:
            theme_info = self.spiritual_knowledge.get_concept_info(theme)
            if theme_info:
                theme_analysis += f"- {theme}: {theme_info.get('definition', '')}\n"
        return theme_analysis
        
    def _analyze_verses(self, article: Dict, verses: List[str]) -> str:
        """Analyze Bible verses in the article."""
        verse_analysis = "The article references and applies these Bible verses:\n"
        for verse in verses:
            verse_analysis += f"- {verse}\n"
        return verse_analysis
        
    def _generate_applications(self, article: Dict) -> str:
        """Generate practical applications from the article."""
        return f"Practical applications from this article include:\n1. Understanding the main message\n2. Applying biblical principles\n3. Living out the teachings"
        
    def save_training_data(self, data: List[Dict], output_dir: str):
        """Save training data to files."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as JSONL format
            output_file = os.path.join(output_dir, 'training_data.jsonl')
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in data:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
                    
            self.logger.info(f"Saved {len(data)} training examples to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving training data: {e}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize data preparator
    website_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'bythestreamwebsite'
    )
    preparator = FineTuningDataPreparator(website_dir)
    
    # Prepare training data
    training_data = preparator.prepare_training_data()
    
    # Save training data
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'fine_tuning'
    )
    preparator.save_training_data(training_data, output_dir) 