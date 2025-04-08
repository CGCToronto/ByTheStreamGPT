import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from pathlib import Path
import json
import numpy as np
import random
from datasets import Dataset
import time
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_small.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 禁用 WandB 在线同步
os.environ["WANDB_MODE"] = "offline"

class TrainingMonitor(TrainerCallback):
    """自定义训练监控回调"""
    def __init__(self):
        self.start_time = time.time()
        self.best_loss = float('inf')
        self.epoch_start_time = None
        self.step_start_time = None
        self.total_steps = 0
        self.completed_steps = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        logging.info("训练开始")
        self.total_steps = state.max_steps
        self.epoch_start_time = time.time()
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        self.completed_steps = state.global_step
        step_time = time.time() - self.step_start_time
        progress = self.completed_steps / self.total_steps * 100
        
        # 计算预计剩余时间
        elapsed_time = time.time() - self.start_time
        avg_time_per_step = elapsed_time / self.completed_steps
        remaining_steps = self.total_steps - self.completed_steps
        eta = remaining_steps * avg_time_per_step
        
        logging.info(f"进度: {progress:.2f}% | 已完成: {self.completed_steps}/{self.total_steps} | "
                    f"预计剩余时间: {eta/60:.1f}分钟 | 当前步耗时: {step_time:.2f}秒")
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            eval_loss = metrics.get("eval_loss", float('inf'))
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                logging.info(f"新的最佳验证损失: {eval_loss:.4f}")
            
            logging.info(f"验证集评估结果: {metrics}")
            
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        logging.info(f"当前轮次完成 | 耗时: {epoch_time/60:.1f}分钟")
        self.epoch_start_time = time.time()

# 自定义回调保存 LoRA 模型
class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_path)
            logging.info(f"保存检查点到: {checkpoint_path}")
            return control

# 设置缓存目录
cache_dir = Path("D:/bythestream/ByTheStreamGPT/fine_tuning/hf_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(cache_dir)
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
os.environ['HF_DATASETS_CACHE'] = str(cache_dir)

def prepare_dataset(tokenizer, data_path='data/training_data_small.json'):
    """准备训练数据集"""
    # 读取处理后的文章数据
    logging.info(f"正在加载训练数据: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        articles_data = json.load(f)
    
    logging.info(f"加载了 {len(articles_data)} 个训练样本")
    
    # 预处理数据，将嵌套的JSON结构转换为文本
    processed_data = []
    for item in articles_data:
        question = item["question"]
        answer = item["answer"]
        
        # 将答案转换为格式化的文本
        answer_text = json.dumps(answer, ensure_ascii=False, indent=2)
        
        # 创建训练样本
        processed_data.append({
            "text": f"""问题：{question}

回答：{answer_text}"""
        })
    
    # 创建数据集
    dataset = Dataset.from_list(processed_data)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=1024,  # 增加序列长度以容纳更长的内容
            return_tensors="pt"
        )
    
    # 对数据集进行编码
    logging.info("正在对数据集进行编码...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        batch_size=100,  # 增加批处理大小以加快处理速度
        num_proc=4  # 增加并行处理数量
    )
    
    # 分割训练集和验证集
    train_val_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    
    logging.info(f"训练集大小: {len(train_val_split['train'])}, 验证集大小: {len(train_val_split['test'])}")
    
    return train_val_split

def train():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"可用GPU内存: {gpu_memory:.2f} GB")
        torch.cuda.empty_cache()
    
    # 加载模型和tokenizer
    logging.info("正在加载模型和 tokenizer...")
    # 考虑使用更小的模型作为基础
    base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # 确保模型处于训练模式
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # 启用输入梯度计算
    
    # 冻结更多层以加速训练
    for name, param in model.named_parameters():
        # 冻结所有非LoRA参数
        if "lora" not in name:
            param.requires_grad = False
        # 只解冻最后几层的参数
        if "layers.23" in name or "layers.24" in name or "layers.25" in name or "layers.26" in name or "layers.27" in name:
            param.requires_grad = True
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 打印基础模型参数
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"\n基础模型参数统计:")
    logging.info(f"总参数数量: {total_params:,}")
    logging.info(f"模型大小: {total_params * 2 / (1024**3):.2f} GB (FP16)")
    
    # 设置 LoRA
    logging.info("\n正在设置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # 减小rank以减少参数量
        lora_alpha=32,  # 调整alpha以平衡性能和效率
        lora_dropout=0.1,  # 增加dropout以防止过拟合
        target_modules=["q_proj", "v_proj"],  # 减少目标模块数量
        bias="none",
        modules_to_save=None
    )
    
    model = get_peft_model(model, lora_config)
    
    # 确保所有LoRA参数都是可训练的
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    
    # 打印LoRA参数统计
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"\nLoRA参数统计:")
    logging.info(f"可训练参数数量: {trainable_params:,}")
    logging.info(f"总参数数量: {total_params:,}")
    logging.info(f"可训练参数占比: {trainable_params/total_params*100:.2f}%")
    logging.info(f"LoRA模型大小: {trainable_params * 2 / (1024**3):.2f} GB (FP16)")
    
    # 准备数据集
    data_path = "data/training_data_small.json"
    tokenized_dataset = prepare_dataset(tokenizer, data_path)
    
    # 打印数据集统计
    train_size = len(tokenized_dataset["train"])
    val_size = len(tokenized_dataset["test"])
    total_size = train_size + val_size
    logging.info(f"\n数据集统计:")
    logging.info(f"训练集大小: {train_size:,}")
    logging.info(f"验证集大小: {val_size:,}")
    logging.info(f"总样本数: {total_size:,}")
    logging.info(f"训练集占比: {train_size/total_size*100:.2f}%")
    logging.info(f"验证集占比: {val_size/total_size*100:.2f}%")
    
    # 设置训练参数 - 针对小数据集优化
    training_args = TrainingArguments(
        output_dir="./results_small",
        num_train_epochs=1,  # 保持1轮
        per_device_train_batch_size=8,  # 增加批次大小
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,  # 减少梯度累积步数
        evaluation_strategy="steps",
        eval_steps=50,  # 更频繁地评估
        save_strategy="steps",
        save_steps=50,  # 更频繁地保存
        learning_rate=5e-4,  # 降低学习率以提高稳定性
        weight_decay=0.01,
        warmup_steps=25,  # 增加预热步数
        logging_steps=10,  # 更频繁地记录日志
        load_best_model_at_end=True,
        report_to="wandb",
        fp16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    
    # 打印训练参数
    logging.info(f"\n训练参数:")
    logging.info(f"训练轮数: {training_args.num_train_epochs}")
    logging.info(f"批次大小: {training_args.per_device_train_batch_size}")
    logging.info(f"梯度累积步数: {training_args.gradient_accumulation_steps}")
    logging.info(f"学习率: {training_args.learning_rate}")
    logging.info(f"权重衰减: {training_args.weight_decay}")
    logging.info(f"预热步数: {training_args.warmup_steps}")
    logging.info(f"评估步数: {training_args.eval_steps}")
    logging.info(f"保存步数: {training_args.save_steps}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[SavePeftModelCallback(), TrainingMonitor(), EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 开始训练
    logging.info("\n开始训练...")
    try:
        trainer.train()
        logging.info("训练完成！")
    except Exception as e:
        logging.error(f"训练过程中出现错误: {str(e)}")
        raise e

if __name__ == "__main__":
    train() 