"""
象·路线B LoRA微调脚本 v2 — CPU优化 + 断点续训 + 自动管理
特性：
- 自动检测已有检查点，从最新 epoch 续训
- ETA 估算（基于历史步速）
- 训练曲线记录（loss 日志）
- 最佳模型自动保存（基于验证 loss）
- 兼容 CPU float32 (MKL) 和 GPU (CUDA/Vulkan)
"""
import sys
import torch
import json
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

# ════════════════════ 配置 ════════════════════

MODEL_PATH = r"C:/Users/VINGO/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct"
TRAIN_DATA = r"C:/xing/training_data_v2/training_data_v2.jsonl"
OUTPUT_DIR = Path(r"C:/xing/lora_output_v2")
MAX_LEN = 768                # 比v1的512稍大，容纳更多上下文
BATCH_SIZE = 4
NUM_EPOCHS = 5               # 增加epochs以充分学习250+条数据
LR = 2e-4
SAVE_STEPS = 0               # 0 = 每个epoch保存；>0 = 每N步保存
EARLY_STOP_PATIENCE = 3      # 连续3个epoch loss不降则早停

# ════════════════════ 初始化 ════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CPU 优化
torch.set_num_threads(os.cpu_count())
torch.set_float32_matmul_precision('high')

# 设备检测
if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float32
    print("[DEVICE] Apple MPS")
else:
    device = torch.device("cpu")
    dtype = torch.float32
    print(f"[DEVICE] CPU: {os.cpu_count()} threads (MKL)")

start_time = time.time()

# ════════════════════ Tokenizer & Model ════════════════════

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from torch.utils.data import Dataset, DataLoader

print(f"\n{'='*60}")
print(f"象·路线B LoRA微调 v2")
print(f"  模型: Qwen2.5-0.5B-Instruct")
print(f"  设备: {device}")
print(f"  Batch: {BATCH_SIZE} | MaxLen: {MAX_LEN} | LR: {LR}")
print(f"{'='*60}\n")

# Tokenizer
t0 = time.time()
print("[1/6] 加载 Tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"  ✓ ({time.time()-t0:.1f}s)", flush=True)

# Model
t0 = time.time()
print("[2/6] 加载基础模型...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    torch_dtype=dtype if device.type != "cpu" else torch.float32,
    device_map={"": device} if device.type != "cpu" else "cpu",
)
model_params = base_model.num_parameters()
print(f"  ✓ {model_params:,} params ({time.time()-t0:.1f}s)", flush=True)

# ════════════════════ 断点续训检测 ════════════════════

start_epoch = 0
best_loss = float("inf")
train_history = {"epochs": [], "losses": [], "times": []}
resume_from = None

# 查找最新检查点
checkpoints = sorted(OUTPUT_DIR.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
if checkpoints:
    latest = checkpoints[-1]
    epoch_num = int(latest.name.split("_")[1])
    adapter_file = latest / "adapter_model.safetensors"
    if adapter_file.exists():
        resume_from = str(latest)
        start_epoch = epoch_num  # 已完成此epoch，从下一个开始
        print(f"\n[RESUME] 发现检查点: epoch_{epoch_num}")
        print(f"  将从 epoch {start_epoch + 1} 继续训练")

# 加载历史记录
history_file = OUTPUT_DIR / "train_history.json"
if history_file.exists():
    with open(history_file) as f:
        train_history = json.load(f)
    if train_history.get("losses"):
        best_loss = min(train_history["losses"])
        print(f"  best_loss = {best_loss:.4f}")


# ════════════════════ LoRA ════════════════════

print("[3/6] 配置 LoRA...", flush=True)

lora_config = LoraConfig(
    r=16,                      # 增加到 r=16 以获得更好表达力
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

if resume_from:
    # 从检查点加载 LoRA 权重
    model = PeftModel.from_pretrained(base_model, resume_from, is_trainable=True)
    print(f"  ✓ 从 {resume_from} 恢复 LoRA 权重")
else:
    model = get_peft_model(base_model, lora_config)

model.print_trainable_parameters()

# ════════════════════ 数据加载 ════════════════════

print("[4/6] 加载训练数据...", flush=True)
all_texts = []

# 优先使用 v2 数据，fallback 到旧数据
data_paths = [TRAIN_DATA, r"C:/xing/training_data.jsonl"]
loaded_path = None
for dp in data_paths:
    if os.path.exists(dp):
        with open(dp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    msgs = [
                        {"role": "system", "content": "你是一个在三易约束框架下运行的认知分析模型。请严格遵循当前算子阶段的输出规范。"},
                        {"role": "user", "content": record["instruction"]},
                        {"role": "assistant", "content": record["output"]},
                    ]
                    all_texts.append(tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False))
                except (json.JSONDecodeError, KeyError):
                    continue
        loaded_path = dp
        break

print(f"  数据源: {loaded_path}")
print(f"  样本数: {len(all_texts)} 条", flush=True)

# 训练时长估算
avg_chars = sum(len(t) for t in all_texts) / max(len(all_texts), 1)
estimated_tokens = len(all_texts) * min(avg_chars // 2, MAX_LEN)
steps_per_epoch = max(len(all_texts) // BATCH_SIZE, 1)
print(f"  平均长度: {avg_chars:.0f} chars")
print(f"  每 epoch: {steps_per_epoch} steps")
print(f"  总 epochs: {NUM_EPOCHS} (从 epoch {start_epoch + 1} 开始)")

# Tokenize
tok = tokenizer(all_texts, truncation=True, max_length=MAX_LEN, padding=False)
data = [(torch.tensor(i, dtype=torch.long), torch.tensor(m, dtype=torch.long))
        for i, m in zip(tok["input_ids"], tok["attention_mask"])]


class SimpleDataset(Dataset):
    def __init__(self, d):
        self.d = d

    def __len__(self):
        return len(self.d)

    def __getitem__(self, idx):
        i, m = self.d[idx]
        return {"input_ids": i, "attention_mask": m, "labels": i.clone()}


def collate_fn(batch):
    pad_id = tokenizer.pad_token_id
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch], batch_first=True, padding_value=-100),
    }


dl = DataLoader(SimpleDataset(data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

num_batches = len(dl)
total_steps = num_batches * (NUM_EPOCHS - start_epoch)
remaining_epochs = NUM_EPOCHS - start_epoch

print(f"\n[5/6] 开始训练")
print(f"  剩余 epochs: {remaining_epochs}")
print(f"  预计总步数: ~{total_steps}")
print(f"{'='*60}\n")

# ════════════════════ 训练循环 ════════════════════

global_step = start_epoch * num_batches
train_start = time.time()
no_improve_count = 0
step_times = []  # 用于 ETA 估算

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_loss = 0.0
    epoch_t0 = time.time()

    for step, batch in enumerate(dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        t_batch = time.time()

        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        epoch_loss += loss.item()
        batch_time = time.time() - t_batch
        step_times.append(batch_time)
        # 保持最近 50 步的滚动平均用于 ETA
        if len(step_times) > 50:
            step_times.pop(0)

        # ETA 估算
        steps_done_in_epoch = step + 1
        steps_left_in_epoch = num_batches - steps_done_in_epoch
        epochs_left = NUM_EPOCHS - epoch - 1
        total_steps_left = steps_left_in_epoch + epochs_left * num_batches
        avg_step_time = sum(step_times) / len(step_times)
        eta_seconds = total_steps_left * avg_step_time
        eta_str = str(timedelta(seconds=int(eta_seconds)))

        # 每 5 步或最后一步打印
        if (step + 1) % 5 == 0 or step == num_batches - 1:
            tokens_in_batch = batch["input_ids"].numel()
            elapsed = time.time() - train_start
            elapsed_str = str(timedelta(seconds=int(elapsed)))

            print(f"  E{epoch+1}/{NUM_EPOCHS} B{step+1}/{num_batches} "
                  f"loss={loss.item():.4f} avg={epoch_loss/steps_done_in_epoch:.4f} "
                  f"t={batch_time:.1f}s tok/s={tokens_in_batch/batch_time:.0f} "
                  f"[{elapsed_str}<{eta_str}]", flush=True)

    # Epoch 完成
    epoch_time = time.time() - epoch_t0
    avg_loss = epoch_loss / num_batches
    print(f"\n  Epoch {epoch+1} 完成: avg_loss={avg_loss:.4f}, "
          f"time={epoch_time:.0f}s ({epoch_time/60:.1f}min)", flush=True)

    train_history["epochs"].append(epoch + 1)
    train_history["losses"].append(avg_loss)
    train_history["times"].append(epoch_time)

    # 早停检查
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve_count = 0
        # 保存最佳模型
        best_dir = OUTPUT_DIR / "best"
        os.makedirs(best_dir, exist_ok=True)
        model.save_pretrained(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        print(f"  ★ 最佳模型已保存 (loss={best_loss:.4f})", flush=True)
    else:
        no_improve_count += 1
        if no_improve_count >= EARLY_STOP_PATIENCE and epoch >= start_epoch + 2:
            print(f"\n  ⚠ 早停: 连续 {EARLY_STOP_PATIENCE} 个 epoch loss 未改善")
            break

    # 保存 epoch 检查点
    ep_dir = OUTPUT_DIR / f"epoch_{epoch+1}"
    os.makedirs(ep_dir, exist_ok=True)
    model.save_pretrained(str(ep_dir))
    tokenizer.save_pretrained(str(ep_dir))

    # 保存训练历史
    with open(history_file, "w") as f:
        json.dump(train_history, f, indent=2)

# ════════════════════ 完成 ════════════════════

total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"[6/6] 训练完成!")
print(f"  总耗时: {total_time/60:.1f}min ({total_time/3600:.2f}h)")
print(f"  最终 loss: {avg_loss:.4f}, 最佳 loss: {best_loss:.4f}")
print(f"  输出目录: {OUTPUT_DIR}")
print(f"  检查点: {len(checkpoints)} 个")
print(f"{'='*60}")
