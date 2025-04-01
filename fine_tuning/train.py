import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import wandb
from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm import tqdm
import json
import numpy as np
import random

# 自定义LoRA模型保存回调
class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_path)
            return control

# 设置Hugging Face缓存目录到D盘
cache_dir = Path("D:/bythestream/ByTheStreamGPT/fine_tuning/hf_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(cache_dir)
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
os.environ['HF_DATASETS_CACHE'] = str(cache_dir)

def download_model():
    print("正在从Ollama下载模型...")
    model_name = "deepseek-r1:1.5b"  # Ollama模型名称
    try:
        # 使用ollama pull命令下载模型
        import subprocess
        subprocess.run(["ollama", "pull", model_name], check=True)
        print("模型下载完成！")
        return model_name
    except Exception as e:
        print(f"模型下载出错: {str(e)}")
        return model_name

def load_model_and_tokenizer():
    print("正在加载模型和tokenizer...")
    base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # 配置LoRA
    print("正在设置LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # 获取PEFT模型
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def setup_lora():
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )
    return lora_config

def prepare_dataset(tokenizer):
    print("正在准备数据集...")
    # 加载数据集
    dataset = load_dataset("csv", data_files="ByTheStreamGPT/fine_tuning/data/training_data.csv")
    
    def tokenize_function(examples):
        # 解析JSON格式的对话
        texts = []
        for text in examples["text"]:
            try:
                conversation = json.loads(text)
                # 获取更多上下文信息
                article_info = conversation[0]["content"]  # 文章信息
                user_question = conversation[1]["content"]
                assistant_answer = conversation[2]["content"]
                
                # 构建更详细的输入文本
                formatted_text = f"""<think>请基于溪水旁杂志的内容回答问题。要求：
1. 回答要深入且专业，体现神学深度
2. 准确引用杂志内容，使用格式：[Volume X, "标题", 作者]
3. 回答要结构清晰，层次分明
4. 对于神学问题，需要从圣经、教义和牧养实践三个层面回答
5. 对于杂志主题，需要准确识别并回应文章的核心观点
6. 回答要体现对杂志整体风格和主题的把握

文章信息：{article_info}

问题：{user_question}

回答：{assistant_answer}</think>"""
                texts.append(formatted_text)
            except Exception as e:
                print(f"处理数据时出错: {str(e)}")
                continue
        
        # 进行分词
        outputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=1024,  # 增加最大长度以容纳更多上下文
            return_tensors="pt"
        )
        
        # 创建标签
        outputs["labels"] = outputs["input_ids"].clone()
        
        return outputs
    
    # 处理数据集
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=16,  # 减小批处理大小以适应更长的序列
        remove_columns=dataset["train"].column_names,
        num_proc=8,  # 增加进程数
        desc="处理数据集"
    )
    
    # 添加数据集信息
    print(f"数据集大小: {len(tokenized_dataset['train'])}")
    # 计算平均序列长度
    lengths = [len(x) for x in tokenized_dataset["train"]["input_ids"]]
    avg_length = sum(lengths) / len(lengths)
    print(f"平均序列长度: {avg_length:.2f}")
    
    return tokenized_dataset

def train():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 加载模型和tokenizer
    print("正在加载基础模型...")
    base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # 加载tokenizer
    print("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # 配置LoRA
    print("正在配置LoRA...")
    lora_config = LoraConfig(
        r=128,               # 增加LoRA秩以提升模型容量
        lora_alpha=256,      # 增加alpha值以增强学习能力
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.15,   # 增加dropout以防止过拟合
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 准备数据集
    print("正在准备数据集...")
    dataset = prepare_dataset(tokenizer)
    
    # 分割训练集和验证集
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=15,              # 增加训练轮数
        per_device_train_batch_size=8,    # 减小批次大小以适应更长的序列
        gradient_accumulation_steps=2,     # 增加梯度累积步数
        learning_rate=2e-5,               # 降低学习率以获得更稳定的训练
        weight_decay=0.1,                 # 增加权重衰减以防止过拟合
        logging_steps=5,                  # 更频繁的日志记录
        save_strategy="steps",
        save_steps=50,                    # 更频繁的模型保存
        warmup_steps=100,                 # 增加预热步数
        max_grad_norm=0.8,                # 降低梯度裁剪阈值以获得更稳定的训练
        lr_scheduler_type="cosine_with_restarts",
        report_to=["tensorboard"],
        evaluation_strategy="steps",
        eval_steps=50,                    # 更频繁的评估
        metric_for_best_model="loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        fp16=True,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        group_by_length=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["input_ids", "attention_mask", "labels"],
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    # 确保模型参数可训练
    model.train()  # 设置为训练模式
    for param in model.parameters():
        param.requires_grad = True
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            ),
            SavePeftModelCallback()
        ]
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    print("保存最终模型...")
    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    
    # 保存训练指标
    print("保存训练指标...")
    metrics = trainer.state.log_history
    with open("./training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print("训练完成！")

if __name__ == "__main__":
    train() 