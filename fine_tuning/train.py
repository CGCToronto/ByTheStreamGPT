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
    dataset = load_dataset("csv", data_files="data/training_data.csv")
    
    def tokenize_function(examples):
        # 解析JSON格式的对话
        texts = []
        for text in examples["text"]:
            try:
                conversation = json.loads(text)
                # 只使用用户问题和助手回答
                user_question = conversation[1]["content"]
                assistant_answer = conversation[2]["content"]
                # 构建输入文本，添加更明确的格式要求
                formatted_text = f"""<think>请基于溪水旁杂志的内容回答问题。要求：
1. 回答要简洁精炼，突出要点
2. 如果引用杂志内容，使用格式：[Volume X, "标题", 作者]
3. 限制回答长度在200字以内
4. 避免重复和冗余表达

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
            max_length=512,
            return_tensors="pt"
        )
        
        # 创建标签
        outputs["labels"] = outputs["input_ids"].clone()
        
        return outputs
    
    # 处理数据集
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=32,  # 增加批处理大小
        remove_columns=dataset["train"].column_names,
        num_proc=4,  # 使用多进程处理
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
        r=32,                # 增加LoRA秩
        lora_alpha=64,       # 增加alpha值
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,    # 增加dropout以减少过拟合
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
        num_train_epochs=15,             # 增加训练轮次
        per_device_train_batch_size=4,   # 增加批次大小
        gradient_accumulation_steps=4,    # 减少梯度累积步数
        learning_rate=2e-5,              # 降低学习率
        weight_decay=0.05,               # 增加权重衰减
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        warmup_steps=50,                 # 减少预热步数
        max_grad_norm=0.5,               # 降低梯度裁剪阈值
        lr_scheduler_type="cosine_with_restarts",  # 使用余弦退火with restarts
        report_to=["tensorboard"],
        evaluation_strategy="steps",      # 添加评估策略
        eval_steps=50,                    # 评估步数
        metric_for_best_model="loss",     # 使用loss作为最佳模型指标
        load_best_model_at_end=True,      # 在训练结束时加载最佳模型
        greater_is_better=False,          # loss越小越好
        bf16=True,                        # 使用bfloat16而不是fp16
        optim="adamw_torch",              # 使用AdamW优化器
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,      # 启用梯度检查点以节省显存
        group_by_length=True,             # 按长度分组以提高效率
        dataloader_num_workers=4,         # 增加数据加载器的工作进程数
        dataloader_pin_memory=True,       # 启用内存固定
        remove_unused_columns=False,      # 防止自动移除未使用的列
        label_names=["input_ids", "attention_mask", "labels"],  # 明确指定标签名称
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