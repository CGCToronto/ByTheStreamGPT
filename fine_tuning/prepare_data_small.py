import json
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
import jieba
import jieba.analyse
import numpy as np
from collections import Counter
import random

# 中文停用词
STOPWORDS = set([
    '的', '了', '和', '是', '就', '都', '而', '及', '与', '这', '那',
    '在', '中', '有', '我', '你', '他', '她', '它', '们', '个', '也',
    '很', '到', '说', '要', '去', '会', '着', '没有', '看', '好', '自己',
    '这', '样', '来', '过', '对', '能', '下', '个', '时', '大', '上',
    '为', '里', '年', '生', '到', '后', '面', '之', '小', '心', '多',
    '经', '发', '现', '所', '成', '用', '家', '种', '事', '作', '方',
    '想', '国', '产', '度', '子', '战', '动', '同', '体', '当', '样',
    '现', '或', '新', '前', '但', '因', '只', '从', '对', '如', '等',
    '还', '其', '将', '两', '本', '已', '组', '无', '设', '建', '使',
    '好', '向', '通', '过', '量', '子', '长', '定', '深', '法', '表',
    '着', '水', '理', '化', '界', '品', '白', '应', '位', '门', '地',
    '合', '明', '四', '实', '期', '反', '关', '各', '内', '相', '平',
    '系', '高', '月', '员', '安', '性', '种', '正', '外', '间', '变',
    '与', '关', '起', '把', '道', '次', '此', '物', '活', '决', '解',
    '又', '或', '意', '比', '先', '手', '见', '只', '主', '么', '公',
    '已', '经', '问', '很', '最', '可', '并', '但', '已', '经', '问',
    '很', '最', '可', '并', '但', '已', '经', '问', '很', '最', '可',
    '并', '但', '已', '经', '问', '很', '最', '可', '并', '但', '已',
    '经', '问', '很', '最', '可', '并', '但', '已', '经', '问', '很',
    '最', '可', '并', '但', '已', '经', '问', '很', '最', '可', '并',
    '但', '已', '经', '问', '很', '最', '可', '并', '但', '已', '经',
    '问', '很', '最', '可', '并', '但', '已', '经', '问', '很', '最',
    '可', '并', '但', '已', '经', '问', '很', '最', '可', '并', '但'
])

# 需要保持原样的关键词
PROTECTED_KEYWORDS = {
    '神', '基督', '耶稣', '圣经', '信仰', '祷告', '教会', 
    '属灵', '成长', '苦难', '见证', '爱', '恩典', '救赎',
    '福音', '圣灵', '十字架', '复活', '永生', '天国', '门徒',
    '使徒', '先知', '祭司', '牧者', '长老', '执事', '圣徒',
    '罪', '悔改', '赦免', '信心', '盼望', '爱心', '喜乐',
    '平安', '忍耐', '温柔', '良善', '信实', '节制', '谦卑',
    '顺服', '奉献', '敬拜', '赞美', '感恩', '代求', '守望',
    '团契', '事工', '宣教', '牧养', '教导', '服事', '见证'
}

# 中文同义词词典（示例）
SYNONYM_DICT = {
    '明白': ['理解', '懂得', '知晓', '了解'],
    '重要': ['关键', '主要', '首要', '重大'],
    '帮助': ['协助', '辅助', '支援', '支持'],
    '成长': ['发展', '进步', '提升', '成熟'],
    '学习': ['研习', '修习', '进修', '学习'],
    '教导': ['教育', '指导', '引导', '训导'],
    '经历': ['体验', '经验', '历程', '过程'],
    '改变': ['转变', '变化', '转换', '变革'],
    '信心': ['信念', '信任', '信赖', '相信'],
    '盼望': ['期望', '期待', '希望', '盼望'],
    '爱心': ['慈爱', '仁爱', '关爱', '爱护'],
    '喜乐': ['快乐', '欢乐', '喜悦', '欢欣'],
    '平安': ['安宁', '平静', '平和', '安稳'],
    '忍耐': ['耐心', '坚忍', '持久', '坚持'],
    '温柔': ['柔和', '温顺', '和善', '温和'],
    '良善': ['善良', '仁慈', '和善', '良善'],
    '信实': ['诚实', '可靠', '可信', '信实'],
    '节制': ['克制', '自律', '约束', '节制'],
    '谦卑': ['谦虚', '谦逊', '谦和', '谦卑'],
    '顺服': ['服从', '顺从', '听从', '顺服'],
    '奉献': ['付出', '献上', '献出', '奉献'],
    '敬拜': ['崇拜', '礼拜', '敬奉', '敬拜'],
    '赞美': ['颂赞', '称颂', '歌颂', '赞美'],
    '感恩': ['感谢', '感激', '谢恩', '感恩'],
    '代求': ['代祷', '代求', '代请', '代求'],
    '守望': ['看守', '守护', '看顾', '守望'],
    '团契': ['团契', '团契', '团契', '团契'],
    '事工': ['工作', '事工', '服事', '事工'],
    '宣教': ['传教', '宣教', '布道', '宣教'],
    '牧养': ['牧养', '牧养', '牧养', '牧养'],
    '教导': ['教导', '教导', '教导', '教导'],
    '服事': ['服事', '服事', '服事', '服事'],
    '见证': ['见证', '见证', '见证', '见证']
}

def get_synonyms(word):
    """获取词语的同义词"""
    if word in PROTECTED_KEYWORDS:
        return [word]
    
    # 从同义词词典中获取
    if word in SYNONYM_DICT:
        return SYNONYM_DICT[word]
    
    # 如果没有找到同义词，返回原词
    return [word]

def replace_with_synonyms(text, ratio=0.3):
    """替换文本中的词语为同义词"""
    # 使用jieba进行分词
    words = list(jieba.cut(text))
    
    # 过滤停用词
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    
    # 计算需要替换的词语数量
    replace_count = max(1, int(len(words) * ratio))
    
    # 随机选择要替换的词语
    replace_indices = random.sample(range(len(words)), min(replace_count, len(words)))
    
    # 替换词语
    for idx in replace_indices:
        word = words[idx]
        if word not in PROTECTED_KEYWORDS:
            synonyms = get_synonyms(word)
            if len(synonyms) > 1:
                words[idx] = random.choice(synonyms)
    
    return ''.join(words)

def transform_sentence(sentence):
    """转换句子结构"""
    # 使用jieba进行分词
    words = list(jieba.cut(sentence))
    
    # 将陈述句转换为疑问句
    if random.random() < 0.3:
        if '是' in words:
            return f"为什么{sentence}？"
        elif '有' in words:
            return f"如何{sentence}？"
        else:
            return f"什么是{sentence}？"
    
    # 添加修饰语
    if random.random() < 0.3:
        modifiers = ['特别', '非常', '极其', '十分', '相当', '格外', '分外', '异常']
        if len(words) > 3:
            insert_pos = random.randint(1, len(words)-1)
            words.insert(insert_pos, random.choice(modifiers))
            return ''.join(words)
    
    return sentence

def expand_context(text):
    """扩展上下文"""
    # 添加可能的背景信息
    if random.random() < 0.3:
        backgrounds = [
            "在属灵生活中，",
            "从信仰的角度来看，",
            "在教会实践中，",
            "在基督徒的成长过程中，",
            "在圣经教导中，",
            "在属灵争战中，",
            "在事奉道路上，",
            "在信仰历程中，"
        ]
        text = random.choice(backgrounds) + text
    
    # 添加解释性内容
    if random.random() < 0.3:
        explanations = [
            "这意味着",
            "这告诉我们",
            "这表明",
            "这启示我们",
            "这提醒我们",
            "这教导我们",
            "这显明",
            "这彰显"
        ]
        text = text + "，" + random.choice(explanations) + "。"
    
    return text

def clean_text(text):
    """清理文本中的特殊符号和占位符"""
    if not isinstance(text, str):
        return ""
    
    # 清理照片占位符
    text = re.sub(r'<[^>]+>', '', text)  # 移除HTML标签
    text = re.sub(r'\[图片\d*\]', '', text)  # 移除[图片]标记
    text = re.sub(r'<[^>]+\.(jpg|jpeg|png|gif)>', '', text)  # 移除图片文件名
    
    # 清理特殊符号
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）《》\s]', '', text)
    
    # 清理多余空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def extract_key_points(article):
    """提取文章的关键信息"""
    key_points = {
        'title': article.get('title', ''),
        'author': article.get('author', ''),
        'volume': article.get('volume', ''),
        'category': article.get('category', ''),
        'overview': '',
        'key_paragraphs': [],
        'core_points': []
    }
    
    # 提取内容
    content_list = article.get('content', [])
    if content_list:
        # 将内容列表转换为字符串
        content = '\n'.join([str(p) for p in content_list if p and p.strip()])
        
        # 清理内容
        content = clean_text(content)
        
        # 提取概述（前200字）
        key_points['overview'] = content[:200] + '...' if len(content) > 200 else content
        
        # 提取关键段落
        paragraphs = content.split('\n')
        paragraph_scores = []
        
        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue
                
            score = 0
            # 1. 长度得分
            if 50 <= len(para) <= 200:
                score += 3
            elif len(para) > 200:
                score += 2
            elif 20 <= len(para) < 50:
                score += 1
                
            # 2. 关键词匹配
            keywords = extract_keywords(content)
            for word in keywords:
                if word in para:
                    score += 2
                    
            # 3. 属灵词汇匹配
            for word in PROTECTED_KEYWORDS:
                if word in para:
                    score += 3
                    
            # 4. 位置权重
            position = i / len(paragraphs)
            if 0.2 <= position <= 0.8:
                score += 2
                
            paragraph_scores.append((para, score))
        
        # 选择得分最高的3个段落
        sorted_paragraphs = sorted(paragraph_scores, key=lambda x: x[1], reverse=True)
        key_points['key_paragraphs'] = [p[0] for p in sorted_paragraphs[:3]]
        
        # 提取核心教导
        for para in key_points['key_paragraphs']:
            if any(word in para for word in PROTECTED_KEYWORDS):
                key_points['core_points'].append(para)
    
    return key_points

def generate_questions(title, content, author):
    """生成问题"""
    questions = []
    
    # 确保输入都是字符串
    title = str(title) if title else ""
    content = str(content) if content else ""
    author = str(author) if author else ""
    
    # 搜索相关的问题
    if title:
        questions.append(f"请搜索关于{title}的文章")
        questions.append(f"查找{title}这篇文章")
        questions.append(f"搜索{title}的内容")
    
    if author:
        questions.append(f"查找{author}发表的文章")
    
    # 内容解释的问题
    if content:
        # 提取关键词和关键句子
        keywords = extract_keywords(content)
        key_sentences = extract_key_sentences(content)
        
        # 确保关键词是字符串
        for keyword in keywords:
            if keyword and isinstance(keyword, str):
                questions.append(f"解释{title}中'{keyword}'的含义")
                questions.append(f"{title}中提到的'{keyword}'是什么意思")
        
        # 确保关键句子是字符串
        for sentence in key_sentences:
            if sentence and isinstance(sentence, str):
                # 限制句子长度
                short_sentence = sentence[:50] + '...' if len(sentence) > 50 else sentence
                questions.append(f"解释{title}中这句话的含义：'{short_sentence}'")
                questions.append(f"请解释{title}中的这句话：'{short_sentence}'")
    
    return questions

def extract_keywords(text):
    """提取文本中的关键词，使用TF-IDF和TextRank结合的方法"""
    # 确保输入是字符串
    if not isinstance(text, str):
        text = str(text) if text else ""
    
    if not text:
        return []
    
    try:
        # 使用jieba的TF-IDF提取关键词
        tfidf_keywords = jieba.analyse.extract_tags(
            text,
            topK=10,
            withWeight=True,
            allowPOS=('n', 'vn', 'v', 'a', 'ad')
        )
        
        # 使用jieba的TextRank提取关键词
        textrank_keywords = jieba.analyse.textrank(
            text,
            topK=10,
            withWeight=True,
            allowPOS=('n', 'vn', 'v', 'a', 'ad')
        )
        
        # 合并两种方法的结果
        keyword_scores = {}
        for word, score in tfidf_keywords:
            if word not in STOPWORDS and len(word) > 1:
                keyword_scores[word] = keyword_scores.get(word, 0) + score
        
        for word, score in textrank_keywords:
            if word not in STOPWORDS and len(word) > 1:
                keyword_scores[word] = keyword_scores.get(word, 0) + score
        
        # 按综合得分排序
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_keywords[:10]]
    except Exception as e:
        print(f"提取关键词时出错: {str(e)}")
        return []

def extract_key_sentences(text):
    """提取文本中的关键句子，使用改进的评分机制"""
    # 确保输入是字符串
    if not isinstance(text, str):
        text = str(text) if text else ""
    
    if not text:
        return []
    
    # 分割句子
    sentences = re.split(r'[。！？]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    # 提取关键词
    keywords = extract_keywords(text)
    
    # 计算每个句子的得分
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        score = 0
        
        # 1. 基础分：句子长度
        length = len(sentence)
        if 20 <= length <= 100:
            score += 5
        elif length > 100:
            score += 3
        elif 10 <= length < 20:
            score += 2
            
        # 2. 关键词匹配
        for keyword in keywords:
            if keyword in sentence:
                score += 3
                
        # 3. 属灵词汇匹配
        for word in PROTECTED_KEYWORDS:
            if word in sentence:
                score += 2
                
        # 4. 位置权重
        position = i / len(sentences)
        if 0.2 <= position <= 0.8:  # 中间段落权重更高
            score += 2
            
        # 5. 句子完整性
        if sentence.endswith(('。', '！', '？')):
            score += 1
            
        sentence_scores.append((sentence, score))
    
    # 按得分排序并选择得分最高的5个句子
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    return [s[0] for s in sorted_sentences[:5]]

def augment_data(question, answer, ratio=0.3, methods=None):
    """数据增强，支持多种增强方法和可配置的参数"""
    if methods is None:
        methods = ['synonym', 'transform', 'expand']
    
    # 确保输入是字符串
    question = str(question) if question else ""
    
    augmented_data = []
    
    # 1. 同义词替换
    if 'synonym' in methods and question:
        try:
            aug_question = replace_with_synonyms(question, ratio=ratio)
            aug_answer = {
                "文章信息": answer["文章信息"],
                "主要内容": {
                    "概述": str(answer["主要内容"]["概述"]) if answer["主要内容"]["概述"] else "",
                    "关键段落": [str(p) for p in answer["主要内容"]["关键段落"] if p]
                },
                "关键词解释": {
                    str(k): str(v) for k, v in answer["关键词解释"].items() if k and v
                },
                "关键句子解释": {
                    str(k): str(v) for k, v in answer["关键句子解释"].items() if k and v
                }
            }
            augmented_data.append({
                "question": aug_question,
                "answer": aug_answer
            })
        except Exception as e:
            print(f"同义词替换时出错: {str(e)}")
    
    # 2. 句式转换
    if 'transform' in methods and question:
        try:
            transformed_question = transform_sentence(question)
            if transformed_question != question:
                augmented_data.append({
                    "question": transformed_question,
                    "answer": answer
                })
        except Exception as e:
            print(f"句式转换时出错: {str(e)}")
    
    # 3. 上下文扩展
    if 'expand' in methods and question:
        try:
            expanded_answer = {
                "文章信息": answer["文章信息"],
                "主要内容": {
                    "概述": expand_context(str(answer["主要内容"]["概述"])) if answer["主要内容"]["概述"] else "",
                    "关键段落": [expand_context(str(p)) for p in answer["主要内容"]["关键段落"] if p]
                },
                "关键词解释": {
                    str(k): expand_context(str(v)) for k, v in answer["关键词解释"].items() if k and v
                },
                "关键句子解释": {
                    str(k): expand_context(str(v)) for k, v in answer["关键句子解释"].items() if k and v
                }
            }
            augmented_data.append({
                "question": question,
                "answer": expanded_answer
            })
        except Exception as e:
            print(f"上下文扩展时出错: {str(e)}")
    
    return augmented_data

def process_article(article):
    """处理单篇文章，生成训练数据"""
    try:
        # 提取关键信息
        key_points = extract_key_points(article)
        
        # 生成问题
        questions = generate_questions(key_points['title'], key_points['overview'], key_points['author'])
        
        # 构建答案
        answer = {
            "文章信息": {
                "标题": key_points['title'],
                "作者": key_points['author'],
                "卷期": key_points['volume'],
                "类别": key_points['category']
            },
            "主要内容": {
                "概述": key_points['overview'],
                "关键段落": key_points['key_paragraphs']
            },
            "关键词解释": {},
            "关键句子解释": {}
        }
        
        # 提取关键词和关键句子
        content_list = article.get('content', [])
        if content_list:
            # 确保所有内容都是字符串
            content = '\n'.join([str(p) for p in content_list if p and str(p).strip()])
            keywords = extract_keywords(content)
            key_sentences = extract_key_sentences(content)
            
            # 生成关键词解释
            for keyword in keywords:
                if keyword not in STOPWORDS and len(keyword) > 1:
                    answer["关键词解释"][keyword] = explain_keyword(keyword, content)
            
            # 生成关键句子解释
            for sentence in key_sentences:
                if len(sentence) > 10:
                    answer["关键句子解释"][sentence] = explain_sentence(sentence, content)
        
        # 生成训练数据
        training_data = []
        for question in questions:
            # 原始数据
            training_data.append({
                "question": question,
                "answer": answer
            })
            
            # 数据增强
            augmented_data = augment_data(question, answer)
            training_data.extend(augmented_data)
        
        return training_data
        
    except Exception as e:
        print(f"处理文章时出错: {str(e)}")
        return None

def explain_keyword(keyword, content):
    """解释关键词在文章中的含义，生成更简洁的解释"""
    # 找到包含关键词的段落
    paragraphs = content.split('\n')
    context_paragraphs = [p for p in paragraphs if keyword in p]
    
    if not context_paragraphs:
        return "未找到相关解释"
    
    # 提取最相关的段落
    context = context_paragraphs[0]
    
    # 使用jieba进行分词
    words = list(jieba.cut(context))
    
    # 提取关键词周围的句子
    sentences = re.split(r'[。！？]', context)
    keyword_sentences = [s for s in sentences if keyword in s]
    
    if not keyword_sentences:
        return context
    
    # 选择最相关的句子
    relevant_sentence = keyword_sentences[0]
    
    # 生成解释
    explanation = f"在文章中，'{keyword}'出现在这样的上下文中：{relevant_sentence}"
    
    # 如果有关键词的同义词，添加说明
    syns = get_synonyms(keyword)
    if len(syns) > 1:
        explanation += f"。相关词语包括：{', '.join(syns[1:3])}"
    
    return explanation

def explain_sentence(sentence, content):
    """解释句子在文章中的含义，生成更结构化的解释"""
    # 找到句子所在的段落
    paragraphs = content.split('\n')
    context_paragraph = None
    for para in paragraphs:
        if sentence in para:
            context_paragraph = para
            break
    
    if not context_paragraph:
        return "未找到相关解释"
    
    # 使用jieba提取关键词
    keywords = jieba.analyse.extract_tags(
        sentence,
        topK=5,
        withWeight=False,
        allowPOS=('n', 'vn', 'v', 'a', 'ad')
    )
    
    # 生成解释
    explanation = {
        "原文": sentence,
        "上下文": context_paragraph,
        "关键词": keywords,
        "解释": f"这句话主要讨论了{', '.join(keywords[:3])}等主题，"
    }
    
    # 添加属灵教导相关的解释
    spiritual_words = [w for w in keywords if w in PROTECTED_KEYWORDS]
    if spiritual_words:
        explanation["解释"] += f"特别强调了{', '.join(spiritual_words)}等属灵教导。"
    
    return explanation

def prepare_training_data():
    """准备训练数据，只处理前20卷"""
    training_data = []
    content_dir = Path("../../ByTheStreamWebsite/public/content")
    error_log = []
    
    # 创建输出目录
    os.makedirs("data", exist_ok=True)
    
    # 获取所有期数目录并排序
    volume_dirs = sorted(list(content_dir.glob("volume_*")), 
                        key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else float('inf'))
    
    # 只处理前20卷
    volume_dirs = volume_dirs[:20]
    
    # 遍历前20期数目录
    for volume_dir in tqdm(volume_dirs, desc="处理期数"):
        try:
            # 遍历该期数下的所有文章
            for article_file in tqdm(list(volume_dir.glob("*_s.json")), desc=f"处理{volume_dir.name}的文章"):
                try:
                    with open(article_file, "r", encoding="utf-8") as f:
                        article_data = json.load(f)
                    
                    # 处理文章
                    article_training_data = process_article(article_data)
                    if article_training_data:
                        training_data.extend(article_training_data)
                        print(f"成功处理文章: {article_data.get('title', '未知标题')}")
                        print(f"生成 {len(article_training_data)} 条训练数据")
                    
                except Exception as e:
                    error_msg = f"处理文件 {article_file} 时出错: {str(e)}"
                    print(error_msg)
                    error_log.append({
                        "file": str(article_file),
                        "error": str(e),
                        "type": "article_processing"
                    })
        
        except Exception as e:
            error_msg = f"处理期数 {volume_dir} 时出错: {str(e)}"
            print(error_msg)
            error_log.append({
                "file": str(volume_dir),
                "error": str(e),
                "type": "volume_processing"
            })
    
    # 保存训练数据
    try:
        output_file = "data/training_data_small.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        print(f"\n成功生成 {len(training_data)} 条训练数据，已保存到 {output_file}")
        
        # 保存错误日志
        if error_log:
            error_file = "data/error_log_small.json"
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_log, f, ensure_ascii=False, indent=2)
            print(f"处理过程中出现 {len(error_log)} 个错误，详情请查看 {error_file}")
        
    except Exception as e:
        print(f"保存数据时出错: {str(e)}")
        error_log.append({
            "file": "training_data_small.json",
            "error": str(e),
            "type": "data_saving"
        })
    
    return training_data, error_log

def test_process_article():
    """测试单篇文章处理"""
    test_file = "../../ByTheStreamWebsite/public/content/volume_1/1_Huangzhiqi_s.json"
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            article = json.load(f)
        
        training_data = process_article(article)
        if training_data:
            print(f"成功处理文章: {article['title']}")
            print(f"生成 {len(training_data)} 条训练数据")
            print("\n示例数据:")
            print(json.dumps(training_data[0], ensure_ascii=False, indent=2))
        else:
            print("处理文章失败")
            
    except Exception as e:
        print(f"测试时出错: {str(e)}")

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("data", exist_ok=True)
    
    # 准备训练数据
    training_data, error_log = prepare_training_data()
    
    # 打印统计信息
    print(f"\n处理完成，共生成 {len(training_data)} 条训练数据")
    print(f"数据已保存到 data/training_data_small.json")
    if error_log:
        print(f"处理过程中出现 {len(error_log)} 个错误，详情请查看 data/error_log_small.json") 