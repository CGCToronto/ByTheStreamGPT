import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import re

def load_model_and_tokenizer():
    print("正在加载基础模型...")
    base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # 检查是否有可用的GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    print("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    try:
        # 加载基础模型
        print("正在加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map="auto",  # 使用GPU
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # 加载LoRA权重
        print("正在加载LoRA权重...")
        model = PeftModel.from_pretrained(model, "./fine_tuned_model")
        print("LoRA权重加载成功！")
        
        return model, tokenizer
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None, None

def generate_response(model, tokenizer, prompt, max_length=512):
    try:
        # 构建输入
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # 确保输入数据在正确的设备上
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.5,      # 降低温度使输出更精确
                top_p=0.7,           # 降低top_p使输出更聚焦
                repetition_penalty=1.5, # 增加重复惩罚
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                num_beams=5,         # 增加beam数量以获得更好的结果
                early_stopping=True,
                no_repeat_ngram_size=4,  # 增加n-gram大小
                length_penalty=0.6    # 降低长度惩罚以获得更简短的回答
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回答部分
        if "回答：" in response:
            response = response.split("回答：")[-1].strip()
        
        # 清理格式
        response = response.replace("<think>", "").replace("</think>", "")
        response = response.replace("<response>", "").replace("</response>", "")
        response = response.replace("<回答>", "").replace("</回答>", "")
        response = response.replace("<主一>", "").replace("<主2>", "")
        response = response.replace("<主日志>", "")
        response = response.replace("---", "")
        response = response.replace("<!--", "").replace("-->", "")
        response = response.replace("<s>", "").replace("</s>", "")
        response = response.replace("<div", "").replace("</div>", "")
        response = response.replace("<span", "").replace("</span>", "")
        response = response.replace("<em>", "").replace("</em>", "")
        response = response.replace("<br>", "\n")
        
        # 移除所有HTML标签
        response = re.sub(r'<[^>]+>', '', response)
        
        # 移除图片引用
        response = "\n".join(line for line in response.split("\n") if not any(img in line for img in ["<img", "<image", ".jpg", ".png"]))
        
        # 清理多余的空行和空格
        response = "\n".join(line.strip() for line in response.split("\n") if line.strip())
        
        return response
    except Exception as e:
        print(f"生成回答时出错: {str(e)}")
        return "生成回答时发生错误，请检查模型和输入。"

def main():
    try:
        # 加载模型和tokenizer
        model, tokenizer = load_model_and_tokenizer()
        if model is None or tokenizer is None:
            print("模型加载失败，程序退出。")
            return
        
        # 测试问题列表
        test_questions = [
            "杨磊在溪水旁杂志上发表过哪些文章？请列举具体期数和标题。",
            "黄志琦牧师在溪水旁杂志上发表过哪些文章？请列举具体期数和标题。",
            "溪水旁杂志的创刊历史是怎样的？请说明创刊时间、创办人和发展历程。",
            "溪水旁杂志的主要内容和特点是什么？",
            "杂志中关于基督徒灵命成长的文章有哪些重要观点？",
            "杂志中关于教会生活的文章主要讨论了哪些方面？"
        ]
        
        # 测试每个问题
        for question in test_questions:
            print("\n" + "="*50)
            print(f"问题：{question}")
            print("-"*50)
            
            # 构建提示
            prompt = f"""<think>
请基于溪水旁杂志的内容简要回答问题：

1. 回答必须简洁，限制在200字以内
2. 如果引用杂志内容，使用格式：[Volume X, "标题", 作者]
3. 突出核心观点，避免重复
4. 使用客观准确的表述

问题：{question}

回答："""
            
            # 生成回答
            response = generate_response(model, tokenizer, prompt)
            print(f"回答：{response}")
            print("="*50)
    except Exception as e:
        print(f"程序运行出错: {str(e)}")

if __name__ == "__main__":
    main() 