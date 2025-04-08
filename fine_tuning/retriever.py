import json
import os
from typing import List, Dict
from pathlib import Path

class ContentRetriever:
    def __init__(self, content_dir: str):
        """初始化内容检索器"""
        self.content_dir = content_dir
        self.article_cache = {}  # 缓存文章内容
        self._build_index()
        
    def _build_index(self):
        """构建文章索引"""
        self.articles = []
        
        # 遍历所有卷宗目录
        for volume_dir in os.listdir(self.content_dir):
            if not volume_dir.startswith('volume_'):
                continue
                
            volume_path = os.path.join(self.content_dir, volume_dir)
            if not os.path.isdir(volume_path):
                continue
                
            # 获取卷宗号
            volume_num = volume_dir.split('_')[1]
            
            # 处理该卷宗下的所有文章
            for file in os.listdir(volume_path):
                if not file.endswith('.json') or file.startswith('table_of_content'):
                    continue
                    
                file_path = os.path.join(volume_path, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        article_data = json.load(f)
                    
                    # 提取文章信息
                    title = article_data.get('title', '')
                    author = article_data.get('author', '')
                    content = article_data.get('content', [])
                    if isinstance(content, list):
                        content = '\n'.join(content)
                    
                    # 提取文章主要内容（去除首尾空白段落）
                    paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
                    if paragraphs:
                        main_content = '\n'.join(paragraphs[1:-1]) if len(paragraphs) > 2 else paragraphs[0]
                    else:
                        main_content = ""
                    
                    # 提取关键段落（较长的段落）
                    key_paragraphs = []
                    for para in paragraphs:
                        if len(para) > 50:  # 只保留较长的段落
                            key_paragraphs.append(para)
                    
                    # 添加到索引
                    self.articles.append({
                        'volume': volume_num,
                        'title': title,
                        'author': author,
                        'main_content': main_content,
                        'key_paragraphs': key_paragraphs[:3],  # 最多取3个关键段落
                        'file_path': file_path
                    })
                    
                    # 缓存文章内容
                    self.article_cache[file_path] = article_data
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相关内容"""
        if not self.articles:
            return []
            
        # 根据查询类型进行搜索
        results = []
        query = query.lower()
        
        # 作者搜索
        if "作者" in query or "谁写的" in query:
            for article in self.articles:
                if article['author'].lower() in query:
                    results.append(article)
        
        # 标题搜索
        elif "《" in query and "》" in query:
            title = query[query.find("《")+1:query.find("》")]
            for article in self.articles:
                if title in article['title']:
                    results.append(article)
        
        # 期数搜索
        elif "第" in query and "期" in query:
            volume = query[query.find("第")+1:query.find("期")]
            for article in self.articles:
                if article['volume'] == volume:
                    results.append(article)
        
        # 内容搜索
        else:
            for article in self.articles:
                if (query in article['title'].lower() or 
                    query in article['main_content'].lower() or 
                    any(query in para.lower() for para in article['key_paragraphs'])):
                    results.append(article)
        
        return results[:top_k]
    
    def get_article_content(self, file_path: str) -> Dict:
        """获取文章完整内容"""
        if file_path in self.article_cache:
            return self.article_cache[file_path]
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
                self.article_cache[file_path] = article_data
                return article_data
        except Exception as e:
            print(f"Error loading article {file_path}: {str(e)}")
            return None

def main():
    """测试检索功能"""
    content_dir = "../../ByTheStreamWebsite/public/content"
    retriever = ContentRetriever(content_dir)
    
    # 测试搜索
    query = "祷告的必要性"
    results = retriever.search(query)
    
    print(f"\n搜索查询: {query}")
    print("\n相关文章:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. 《{result['title']}》")
        print(f"   作者：{result['author']}")
        print(f"   期数：第{result['volume']}期")
        print(f"   主要内容：{result['main_content'][:200]}...")
        print(f"   关键段落：")
        for para in result['key_paragraphs']:
            print(f"     - {para}")

if __name__ == "__main__":
    main() 