import json
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def load_volume_list(content_dir):
    with open(os.path.join(content_dir, 'volume_list.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

def process_article(article_data, lang_version):
    # 检查必要的字段是否存在
    if not isinstance(article_data, dict):
        print(f"Warning: Invalid article format: {str(article_data)[:100]}...")
        return None

    # 从文章内容构建问答对
    title = article_data.get('title', '')
    content = article_data.get('content', [])
    category = article_data.get('category', '')
    
    if not title or not content:
        return None
        
    # 将内容列表合并成文本
    content_text = "\n".join(content) if isinstance(content, list) else str(content)
    
    # 构建系统提示词，强调神学知识和牧师的回答风格
    system_prompt = """你是一位经验丰富的基督教牧师，具有深厚的神学知识和丰富的牧养经验。你的回答应该：
1. 基于圣经教导和神学理论
2. 体现牧者的关怀和同理心
3. 使用温和、富有智慧的语言
4. 在回答中融入相关的圣经经文
5. 注重属灵生命的成长
6. 保持谦卑和谨慎的态度
7. 在必要时提供实际的属灵建议

请以牧师的视角回答以下问题："""

    try:
        # 构建用户问题
        user_question = f'请以基督教牧师的视角，解析这篇关于"{title}"的{category}文章，并总结其中的属灵教导。'
        
        # 构建助手回答
        assistant_answer = f"""作为一位牧师，让我从属灵的角度来分享这篇文章的见解：

标题：{title}
类别：{category}

文章内容分析：
{content_text}

属灵教导与反思：
1. 这篇文章展现了基督徒在信仰生活中的重要见证
2. 从中我们可以学习到宝贵的属灵功课
3. 这些经历提醒我们要持守信仰，追求属灵成长

愿这篇文章能够激励我们更深地认识神，活出基督的样式。"""
        
        # 构建对话格式
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": assistant_answer}
        ]
        
        return {
            "text": json.dumps(conversation, ensure_ascii=False),
            "lang_version": lang_version
        }
    except Exception as e:
        print(f"Error processing article: {str(e)}")
        return None

def prepare_training_data(content_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load volume list
    volume_list = load_volume_list(content_dir)
    
    # Process all articles
    all_articles = []
    
    for volume in tqdm(volume_list['volume_list'], desc="Processing volumes"):
        volume_dir = os.path.join(content_dir, volume['folder'])
        
        # Skip if directory doesn't exist
        if not os.path.exists(volume_dir):
            print(f"Warning: Directory not found: {volume_dir}")
            continue
            
        # Process all JSON files in the volume directory
        for file in os.listdir(volume_dir):
            if file.endswith('.json') and not file.startswith('table_of_content') and not file == 'sitemap.xml':
                file_path = os.path.join(volume_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        article_data = json.load(f)
                    
                    # 确定语言版本
                    file_name = Path(file).stem
                    lang_version = "zh" if file_name.endswith("_zh") else "en"
                    
                    # 处理文章
                    processed = process_article(article_data, lang_version)
                    if processed:
                        all_articles.append(processed)
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    if not all_articles:
        raise ValueError("No articles were successfully processed!")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_articles)
    
    # Print statistics
    print(f"\nTotal articles processed: {len(df)}")
    print(f"Language version distribution:\n{df['lang_version'].value_counts()}")
    
    # Save as CSV
    output_file = os.path.join(output_dir, 'training_data.csv')
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Data saved to: {output_file}")

if __name__ == "__main__":
    # 使用绝对路径
    content_dir = "../../ByTheStreamWebsite/public/content"
    output_dir = "data"
    prepare_training_data(content_dir, output_dir) 