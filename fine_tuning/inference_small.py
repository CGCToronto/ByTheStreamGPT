import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
from pathlib import Path
import re
import jieba

def load_model_and_tokenizer():
    """加载模型和分词器"""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            trust_remote_code=True
        )
        
        # 加载LoRA权重 - 使用小数据集训练的模型
        model = PeftModel.from_pretrained(model, "results_small/checkpoint-100")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def extract_keywords(question):
    """从问题中提取关键词"""
    # 使用jieba提取关键词
    words = jieba.cut(question)
    keywords = []
    
    # 定义停用词
    stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    
    for word in words:
        # 过滤停用词和单字
        if len(word) > 1 and word not in stopwords:
            keywords.append(word)
    
    # 如果没有提取到关键词，返回原始问题中的主要词
    if not keywords:
        words = question.split()
        keywords = [w for w in words if len(w) > 1]
    
    return keywords

def find_relevant_articles(articles, question):
    """根据问题找到相关文章"""
    keywords = extract_keywords(question)
    relevant_articles = []
    seen_titles = set()  # 用于去重
    
    for article in articles:
        # 跳过重复标题的文章
        if article['title'] in seen_titles:
            continue
        seen_titles.add(article['title'])
        
        score = 0
        title = article.get('title', '')
        content = article.get('content', '')
        overview = article.get('overview', '')
        category = article.get('category', '')
        author = article.get('author', '')  # 获取作者信息
        
        # 计算匹配分数
        for keyword in keywords:
            # 作者匹配权重最高
            if keyword in author:
                score += 6
            # 标题匹配权重次之
            if keyword in title:
                score += 5
            # 类别匹配权重中等
            if keyword in category:
                score += 4
            # 概述匹配权重较低
            if keyword in overview:
                score += 3
            # 内容匹配权重最低
            if keyword in content:
                score += 1
        
        # 如果分数大于0，说明有相关文章
        if score > 0:
            article['relevance_score'] = score
            relevant_articles.append(article)
    
    # 按相关度排序
    relevant_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
    return relevant_articles[:3]  # 返回最相关的3篇文章

def generate_response(model, tokenizer, prompt, relevant_articles):
    """生成回答"""
    # 构建上下文
    context_text = "找到以下相关文章：\n\n"
    for i, article in enumerate(relevant_articles, 1):
        context_text += f"文章{i}：\n"
        context_text += f"标题：{article['title']}\n"
        context_text += f"作者：{article.get('author', '未知')}\n"  # 显示作者信息
        context_text += f"期数：第{article['volume'].replace('volume_', '')}期\n"
        context_text += f"类别：{article.get('category', '')}\n"
        context_text += f"概述：{article.get('overview', '')}\n"
        if article.get('content'):
            # 只取内容的前200个字符作为参考
            context_text += f"内容摘要：{article['content'][:200]}...\n"
        if article.get('key_paragraphs'):
            context_text += "关键段落：\n"
            for para in article['key_paragraphs']:
                context_text += f"- {para}\n"
        context_text += "\n"

    # 构建提示词
    structured_prompt = f"""请根据以下文章信息，准确回答问题。要求：
1. 只使用中文回答
2. 只使用提供的文章信息
3. 如果信息不足，请明确说明"抱歉，在提供的文章中没有找到相关信息"
4. 保持简洁准确，避免重复
5. 严格按照指定格式回答

{context_text}

问题：{prompt}

请按照以下格式回答，对每篇相关文章分别说明：

第[X]篇文章：
- 标题：[完整标题]
- 作者：[作者姓名]
- 期数：[期数]
- 主要内容：[核心内容概述，100字以内]
- 属灵教导：[属灵教导要点，如有]
- 相关经文：[引用的经文，如有]

回答："""

    # 生成回答
    inputs = tokenizer(structured_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=800,
        temperature=0.5,  # 进一步降低温度
        top_p=0.85,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.4,  # 增加重复惩罚
        no_repeat_ngram_size=4,  # 增加不重复n-gram大小
        num_beams=4,  # 使用beam search
        early_stopping=True
    )
    
    # 提取回答部分（移除提示词）
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("回答：")[-1].strip()
    
    # 清理和格式化回答
    def clean_response(text):
        # 移除英文内容
        text = re.sub(r'[a-zA-Z]+', '', text)
        # 移除URL
        text = re.sub(r'http\S+', '', text)
        # 移除多余的标点
        text = re.sub(r'[,.。，]{2,}', '。', text)
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # 处理回答
    lines = response.split('\n')
    formatted_response = []
    current_article = None
    
    for line in lines:
        line = clean_response(line)
        if not line:
            continue
            
        # 检查是否是新文章的开始
        if line.startswith('第') and '篇文章' in line:
            current_article = line
            formatted_response.append(f"\n{line}：")
            continue
            
        # 处理文章内容
        if line.startswith('-') or line.startswith('•'):
            line = '- ' + line.lstrip('-•').strip()
            if current_article:
                formatted_response.append(line)
                
    # 如果没有找到相关信息
    if not formatted_response:
        return "抱歉，在提供的文章中没有找到相关信息。"
        
    return '\n'.join(formatted_response)

def load_articles():
    """加载文章数据 - 只加载前10卷"""
    articles = []
    content_dir = Path("../../ByTheStreamWebsite/public/content")
    
    # 获取所有期数目录并排序
    volume_dirs = sorted(list(content_dir.glob("volume_*")), 
                        key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else float('inf'))
    
    # 只处理前10卷
    volume_dirs = volume_dirs[:10]
    
    # 遍历前10期数目录
    for volume_dir in volume_dirs:
        volume = volume_dir.name.replace("volume_", "")
        
        # 遍历该期数下的所有文章
        for article_file in volume_dir.glob("*_s.json"):
            try:
                with open(article_file, "r", encoding="utf-8") as f:
                    article_data = json.load(f)
                    
                # 提取文章信息
                article = {
                    "title": article_data.get("title", ""),
                    "author": article_data.get("author", "未知"),  # 获取作者信息
                    "volume": volume,
                    "category": article_data.get("category", ""),
                    "overview": article_data.get("overview", ""),
                    "content": " ".join(article_data.get("content", [])),  # 合并内容段落
                    "key_paragraphs": article_data.get("key_paragraphs", [])
                }
                
                articles.append(article)
            except Exception as e:
                print(f"Error loading article {article_file}: {str(e)}")
    
    return articles

def main():
    # 加载模型和分词器
    print("\n" + "="*80)
    print("正在加载模型和分词器...")
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        return
    print("模型加载完成！")
    print("="*80)

    # 加载文章数据
    print("\n正在加载文章数据...")
    articles = load_articles()
    print(f"成功加载 {len(articles)} 篇文章（前10卷）")
    print("="*80)

    # 测试问题列表 - 针对前10卷的问题
    test_questions = [
        "黄智奇在溪水旁杂志上发表过哪些文章？请列举具体期数和标题。",
        "溪水旁杂志中关于祷告的文章有哪些？",
        "《心灵祷告》（刊首语）中关于祷告的必要性是如何解释的？",
        "溪水旁杂志的创刊号是什么时候出版的？",
        "《主啊，请你差遣我》这篇文章中关于服事的观点是什么？",
        "《祷告有必要吗？》这篇文章中提到的祷告误区有哪些？",
        "溪水旁杂志第9期中关于教会生活的文章有哪些？",
        "《在瓦器中有至宝》这篇文章中提到的'瓦器'比喻是什么意思？"
    ]

    # 对每个问题进行测试
    for i, question in enumerate(test_questions, 1):
        print("\n" + "="*80)
        print(f"问题 {i}/{len(test_questions)}: {question}")
        print("-"*80)
        
        # 找到相关文章
        print("正在检索相关文章...")
        relevant_articles = find_relevant_articles(articles, question)
        print(f"找到 {len(relevant_articles)} 篇相关文章")
        
        # 生成回答
        print("\n正在生成回答...")
        response = generate_response(model, tokenizer, question, relevant_articles)
        
        # 格式化输出
        print("\n回答：")
        print("-"*80)
        print(response)
        print("="*80)
        
        # 添加分隔符，等待用户确认继续
        if i < len(test_questions):
            input("\n按回车键继续测试下一个问题...")

if __name__ == "__main__":
    main() 