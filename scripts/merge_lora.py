"""
合并 LoRA 权重到基础模型，输出完整的 HuggingFace 格式模型。
路线B 验证第一步：将 lora_output_v2/best 合并为独立模型。
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

MODEL_PATH = r"C:/Users/VINGO/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct"
LORA_PATH = r"C:/xing/lora_output_v2/best"
OUTPUT_DIR = Path(r"C:/xing/models/xiang-routeb-0.5b-merged")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  路线B — LoRA 合并")
print("=" * 60)

# 1. 加载基础模型
print(f"\n[1/3] 加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"  ✓ 基础模型加载完成")

# 2. 加载 LoRA 并合并
print(f"\n[2/3] 加载并合并 LoRA: {LORA_PATH}")
lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)
merged_model = lora_model.merge_and_unload()
print(f"  ✓ LoRA 已合并到基础模型")

# 3. 保存
print(f"\n[3/3] 保存合并模型到: {OUTPUT_DIR}")
merged_model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.safetensors")) / (1024 * 1024)
print(f"  ✓ 保存完成 ({size_mb:.0f} MB)")

# 验证
print(f"\n{'=' * 60}")
print(f"  合并完成！")
print(f"  输出: {OUTPUT_DIR}")
print(f"  文件数: {len(list(OUTPUT_DIR.glob('*')))}")
print(f"  下一步: convert_hf_to_gguf.py")
print(f"{'=' * 60}")
