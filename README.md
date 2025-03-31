# 溪水旁杂志模型微调项目

## 项目概述
本项目使用DeepSeek-R1-Distill-Qwen-1.5B模型，通过LoRA方法对溪水旁杂志内容进行微调，以生成与杂志内容相关的回答。

## 数据准备

### 1. 数据收集
1. 从溪水旁杂志网站收集文章内容
2. 将文章保存为JSON格式，包含以下字段：
   ```json
   {
     "title": "文章标题",
     "author": "作者",
     "volume": "期数",
     "content": "文章内容",
     "date": "发布日期"
   }
   ```

### 2. 数据预处理
1. 运行数据预处理脚本：
   ```bash
   python prepare_data.py
   ```
2. 脚本会：
   - 清理HTML标签
   - 标准化文本格式
   - 生成训练所需的问答对
   - 创建训练集和验证集

### 3. 数据格式
处理后的数据格式如下：
```json
{
  "instruction": "请根据溪水旁杂志的内容回答问题。回答要简洁，并注明引用来源。",
  "input": "问题内容",
  "output": "根据[Volume X, '标题', 作者]的内容，回答...",
  "history": []
}
```

## 模型训练

### 1. 环境配置
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 确保GPU环境：
   - CUDA 11.8+
   - PyTorch 2.0+
   - 至少12GB显存

### 2. 训练配置
当前使用的LoRA配置：
```python
lora_config = LoraConfig(
    r=32,                # LoRA秩
    lora_alpha=64,       # LoRA alpha值
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 3. 开始训练
```bash
python train.py
```

## 模型使用

### 1. 本地部署
1. 加载模型和LoRA权重：
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from peft import PeftModel
   
   # 加载基础模型
   model = AutoModelForCausalLM.from_pretrained(
       "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
       trust_remote_code=True,
       device_map="auto",
       torch_dtype=torch.float16
   )
   
   # 加载LoRA权重
   model = PeftModel.from_pretrained(model, "./fine_tuned_model")
   tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
   ```

2. 生成回答：
   ```python
   def generate_response(prompt, model, tokenizer):
       inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
       outputs = model.generate(
           **inputs,
           max_length=512,
           temperature=0.6,
           top_p=0.85,
           repetition_penalty=1.3,
           num_beams=3,
           length_penalty=0.8,
           no_repeat_ngram_size=3
       )
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)
       return response
   ```

### 2. 使用示例
```python
# 测试问题
test_questions = [
    "杨磊在溪水旁杂志上发表过哪些文章？",
    "黄志琦牧师在溪水旁杂志上发表过哪些文章？",
    "溪水旁杂志的创刊历史是怎样的？"
]

# 生成回答
for question in test_questions:
    prompt = f"<think>请根据溪水旁杂志的内容回答问题。回答要简洁，并注明引用来源。</think>\n\n问题：{question}\n\n回答："
    response = generate_response(prompt, model, tokenizer)
    print(f"问题：{question}")
    print(f"回答：{response}\n")
```

## 注意事项
1. 确保GPU内存充足（建议至少12GB）
2. 训练过程中会定期保存检查点
3. 使用早停机制避免过拟合
4. 训练完成后会自动保存最佳模型

## 当前状态
- 训练正在进行中
- 使用早停机制监控训练进度
- 每50步保存一次检查点
- 使用TensorBoard记录训练过程

## 后续计划
1. 评估模型性能
2. 优化生成参数
3. 进行更多测试用例验证
4. 根据反馈进行模型调整 