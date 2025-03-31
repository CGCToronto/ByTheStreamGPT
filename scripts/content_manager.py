import os
import json
from typing import List, Dict
import logging
from datetime import datetime
import hashlib
import re

class ContentManager:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.logger = logging.getLogger(__name__)
        self.articles_cache = {}
        self.last_update = None
        self.max_cache_size = 100  # Maximum number of articles to cache
        self.max_content_length = 500  # Maximum content length to store
        
        # Initialize search indices
        self.author_index = {}  # author -> [article_ids]
        self.verse_index = {}   # verse -> [article_ids]
        self.theme_index = {}   # theme -> [article_ids]
        
    def _truncate_content(self, content: str) -> str:
        """Truncate content to save memory."""
        if len(content) > self.max_content_length:
            return content[:self.max_content_length] + "..."
        return content
        
    def _get_content_hash(self, content: str) -> str:
        """Generate a hash for content to detect changes."""
        return hashlib.md5(content.encode()).hexdigest()
        
    def _extract_verses(self, content: str) -> List[str]:
        """Extract Bible verses from content."""
        # Common Bible verse patterns
        patterns = [
            r'(\d?\s*[A-Za-z]+)\s+(\d+):(\d+)',  # John 3:16
            r'(\d?\s*[A-Za-z]+)\s+(\d+)',         # John 3
            r'(\d?\s*[A-Za-z]+)\s+(\d+):(\d+)-(\d+)',  # John 3:16-17
        ]
        
        verses = []
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                verses.append(match.group(0))
        return verses
        
    def _extract_themes(self, content: str) -> List[str]:
        """Extract spiritual themes from content."""
        # Common spiritual themes
        themes = [
            "redemption", "peace", "love", "faith", "hope", "grace",
            "forgiveness", "salvation", "obedience", "discipleship",
            "prayer", "worship", "fellowship", "service", "mission",
            "evangelism", "discipleship", "stewardship", "wisdom",
            "justice", "mercy", "compassion", "humility", "joy"
        ]
        
        found_themes = []
        content_lower = content.lower()
        for theme in themes:
            if theme in content_lower:
                found_themes.append(theme)
        return found_themes
        
    def load_articles(self, volume: str) -> List[Dict]:
        """Load articles from a specific volume with memory optimization."""
        try:
            volume_path = os.path.join(self.base_dir, f"volume_{volume}")
            if not os.path.exists(volume_path):
                self.logger.warning(f"Volume {volume} not found")
                return []
                
            articles = []
            for file in os.listdir(volume_path):
                if file.endswith('.json'):
                    with open(os.path.join(volume_path, file), 'r', encoding='utf-8') as f:
                        article = json.load(f)
                        # Only store essential information
                        article_data = {
                            'id': article.get('id'),
                            'title': article.get('title'),
                            'volume': article.get('volume'),
                            'content': self._truncate_content(article.get('content', '')),
                            'content_hash': self._get_content_hash(article.get('content', '')),
                            'file_path': os.path.join(volume_path, file),
                            'author': article.get('author', ''),
                            'verses': self._extract_verses(article.get('content', '')),
                            'themes': self._extract_themes(article.get('content', ''))
                        }
                        articles.append(article_data)
                        
                        # Update search indices
                        if article_data['author']:
                            if article_data['author'] not in self.author_index:
                                self.author_index[article_data['author']] = []
                            self.author_index[article_data['author']].append(article_data['id'])
                            
                        for verse in article_data['verses']:
                            if verse not in self.verse_index:
                                self.verse_index[verse] = []
                            self.verse_index[verse].append(article_data['id'])
                            
                        for theme in article_data['themes']:
                            if theme not in self.theme_index:
                                self.theme_index[theme] = []
                            self.theme_index[theme].append(article_data['id'])
            
            return articles
        except Exception as e:
            self.logger.error(f"Error loading volume {volume}: {e}")
            return []
            
    def search_by_author(self, author: str) -> List[Dict]:
        """Search articles by author."""
        try:
            if author not in self.author_index:
                return []
                
            article_ids = self.author_index[author]
            articles = []
            
            for volume, volume_articles in self.articles_cache.items():
                for article in volume_articles:
                    if article['id'] in article_ids:
                        articles.append(article)
                        
            return articles
        except Exception as e:
            self.logger.error(f"Error searching by author: {e}")
            return []
            
    def search_by_verse(self, verse: str) -> List[Dict]:
        """Search articles containing specific Bible verses."""
        try:
            if verse not in self.verse_index:
                return []
                
            article_ids = self.verse_index[verse]
            articles = []
            
            for volume, volume_articles in self.articles_cache.items():
                for article in volume_articles:
                    if article['id'] in article_ids:
                        articles.append(article)
                        
            return articles
        except Exception as e:
            self.logger.error(f"Error searching by verse: {e}")
            return []
            
    def search_by_theme(self, theme: str) -> List[Dict]:
        """Search articles by spiritual theme."""
        try:
            if theme not in self.theme_index:
                return []
                
            article_ids = self.theme_index[theme]
            articles = []
            
            for volume, volume_articles in self.articles_cache.items():
                for article in volume_articles:
                    if article['id'] in article_ids:
                        articles.append(article)
                        
            return articles
        except Exception as e:
            self.logger.error(f"Error searching by theme: {e}")
            return []
            
    def update_cache(self):
        """Update the articles cache with memory optimization."""
        try:
            # Get all volume directories
            volumes = [d for d in os.listdir(self.base_dir) if d.startswith('volume_')]
            
            # Clear old cache if it's too large
            if len(self.articles_cache) > self.max_cache_size:
                self.articles_cache.clear()
            
            # Load articles from each volume
            for volume in volumes:
                volume_num = volume.split('_')[1]
                articles = self.load_articles(volume_num)
                self.articles_cache[volume_num] = articles
                
            self.last_update = datetime.now()
            self.logger.info(f"Cache updated at {self.last_update}")
            
        except Exception as e:
            self.logger.error(f"Error updating cache: {e}")
            
    def get_relevant_articles(self, query: str, volume: str = None) -> List[Dict]:
        """Get relevant articles based on query and volume with memory optimization."""
        try:
            # Update cache if needed
            if not self.last_update or (datetime.now() - self.last_update).days > 1:
                self.update_cache()
                
            # Check for specialized queries
            if "written by" in query.lower():
                author = query.lower().split("written by")[-1].strip()
                return self.search_by_author(author)
                
            # Check for verse queries
            if any(verse in query.lower() for verse in ["verse", "chapter", "book"]):
                verses = self._extract_verses(query)
                if verses:
                    articles = []
                    for verse in verses:
                        articles.extend(self.search_by_verse(verse))
                    return articles
                    
            # Check for theme queries
            themes = self._extract_themes(query)
            if themes:
                articles = []
                for theme in themes:
                    articles.extend(self.search_by_theme(theme))
                return articles
                
            # If no specialized query, use regular search
            if volume:
                articles = self.articles_cache.get(volume, [])
            else:
                articles = []
                for vol_articles in self.articles_cache.values():
                    articles.extend(vol_articles)
                    
            # Simple keyword matching
            relevant_articles = []
            query_keywords = set(query.lower().split())
            
            for article in articles:
                content = article.get('content', '').lower()
                if any(keyword in content for keyword in query_keywords):
                    relevant_articles.append(article)
                    
            return relevant_articles[:3]  # Return top 3 matches to save memory
            
        except Exception as e:
            self.logger.error(f"Error getting relevant articles: {e}")
            return []
            
    def get_article_by_id(self, article_id: str) -> Dict:
        """Get a specific article by ID with lazy loading."""
        try:
            # Search through all volumes
            for volume, articles in self.articles_cache.items():
                for article in articles:
                    if article.get('id') == article_id:
                        # Load full content only when needed
                        if 'file_path' in article:
                            with open(article['file_path'], 'r', encoding='utf-8') as f:
                                full_article = json.load(f)
                                return full_article
            return None
        except Exception as e:
            self.logger.error(f"Error getting article {article_id}: {e}")
            return None

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the content manager
    manager = ContentManager("path/to/bythestreamwebsite")
    manager.update_cache()
    
    # Test different types of queries
    queries = [
        "articles written by John Smith",
        "articles about John 3:16",
        "articles about redemption and peace"
    ]
    
    for query in queries:
        print(f"\nSearching for: {query}")
        articles = manager.get_relevant_articles(query)
        print(f"Found {len(articles)} relevant articles")
        for article in articles:
            print(f"- {article.get('title')} (Volume {article.get('volume')})") 