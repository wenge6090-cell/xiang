"""
验证LoRA微调效果：测试模型在三易约束下的生成表现。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = r"C:/Users/VINGO/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct"
LORA_PATH = r"C:/xing/lora_output"

TEST_PROMPTS = [
    # 测试1：有约束提示的标准任务
    {
        "name": "有约束-项目管理",
        "system": "你是一个在三易约束框架下运行的认知分析模型。",
        "user": """你运行在三易约束框架中。系统通过以下引擎引导你的分析：

【三易约束体系说明】
你运行在一个名为「三易」的认知约束框架中。三易由三台状态机组成。

一、归藏引擎 — 意识循环（八气算子）：生（起念探索）、动（发散联想）、长（收敛聚焦）、育（方案分解）
二、周易引擎 — 认知姿态（八卦体系）
三、连山引擎 — 障碍导航
四、阴仪协议：正则规则的形式检查

当前轮次：
在项目管理领域中，请采用系统化方法解决此核心挑战：如何在资源受限的情况下实现高性能和高可靠性？
步骤1-问题界定：精确定义问题的边界、约束条件和成功标准

每轮完成一个步骤。完成时输出"【步骤1完成】"。

在三易约束框架的引导下完成本轮分析。"""
    },
    # 测试2：无约束的普通对话
    {
        "name": "无约束-普通对话",
        "system": "你是一个有用的AI助手。",
        "user": "请解释什么是项目管理中的资源约束？"
    },
]

print("=" * 60)
print("藏海·约束理解微调 — 验证")
print("=" * 60)

# 加载基础模型
print(f"\n加载基础模型: {MODEL_PATH}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map="cpu",
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True, use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token

# 加载LoRA权重
print(f"加载LoRA权重: {LORA_PATH}")
try:
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    print("LoRA加载成功!")
    HAS_LORA = True
except Exception as e:
    print(f"LoRA加载失败: {e}")
    print("将使用基础模型进行测试")
    model = base_model
    HAS_LORA = False

model.eval()

# 测试
for test in TEST_PROMPTS:
    print(f"\n{'─' * 60}")
    print(f"测试: {test['name']}")
    print(f"{'─' * 60}")
    
    messages = [
        {"role": "system", "content": test["system"]},
        {"role": "user", "content": test["user"]},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    print(f"\n模型回复:\n{response[:500]}")
    print(f"\n--- 长度: {len(response)} chars ---")

print(f"\n{'=' * 60}")
if HAS_LORA:
    print("验证完成 (使用LoRA微调模型)")
else:
    print("验证完成 (使用基础模型 - 未找到LoRA权重)")
print(f"{'=' * 60}")
