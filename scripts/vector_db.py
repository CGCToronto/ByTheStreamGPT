import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config.config import VECTOR_DB_DIR, COLLECTION_NAME

class VectorDBHandler:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            persist_directory=str(VECTOR_DB_DIR),
            anonymized_telemetry=False
        ))
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts."""
        return self.model.encode(texts).tolist()

    def add_articles(self, articles: List[Dict]):
        """Add articles to the vector database."""
        texts = [article['content'] for article in articles]
        embeddings = self.create_embeddings(texts)
        
        # Prepare metadata
        metadatas = []
        ids = []
        for i, article in enumerate(articles):
            metadata = {
                'volume': article['volume'],
                'article_number': article['article_number'],
                'author': article['author'],
                'language': article['language'],
                'theme': article['theme'],
                'theme_english': article['theme_english'],
                'year': article['year'],
                'month': article['month']
            }
            metadatas.append(metadata)
            ids.append(f"{article['volume']}_{article['article_number']}_{article['language']}")

        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query: str, n_results: int = 5, language: str = None) -> List[Dict]:
        """Search for similar articles."""
        query_embedding = self.create_embeddings([query])[0]
        
        # Prepare where clause for language filtering if specified
        where = {"language": language} if language else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results

    def get_article_by_id(self, article_id: str) -> Dict:
        """Retrieve a specific article by ID."""
        result = self.collection.get(ids=[article_id])
        if not result['ids']:
            return None
        
        return {
            'id': result['ids'][0],
            'content': result['documents'][0],
            'metadata': result['metadatas'][0]
        }

    def delete_collection(self):
        """Delete the collection (useful for resetting the database)."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

if __name__ == "__main__":
    # Example usage
    db = VectorDBHandler()
    # Add some test articles
    test_articles = [
        {
            'content': 'This is a test article in English.',
            'volume': '1',
            'article_number': '1',
            'author': 'test',
            'language': 'simplified',
            'theme': 'Test Theme',
            'theme_english': 'Test Theme',
            'year': '2024',
            'month': '1'
        }
    ]
    db.add_articles(test_articles)
    
    # Test search
    results = db.search('test article')
    print(results) 