"""
藏海·约束理解微调 — LoRA 微调脚本 (CPU优化版)
float32 + MKL多线程优化
"""
import sys, torch, json, os, time

# ── MKL多线程优化 ──
torch.set_num_threads(os.cpu_count())
torch.set_float32_matmul_precision('high')

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = r"C:/Users/VINGO/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct"
TRAIN_DATA = r"C:/xing/training_data.jsonl"
OUTPUT_DIR = r"C:/xing/lora_output"
MAX_LEN = 512
BATCH_SIZE = 4          # 增大batch提升MKL利用率
NUM_EPOCHS = 3
LR = 2e-4

device = torch.device("cpu")
print(f"PyTorch: {torch.__version__}, threads={torch.get_num_threads()}, batch={BATCH_SIZE}", flush=True)

# ── Tokenizer ──
print("[1/7] 加载 tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Model (float32 让 MKL 全速运行) ──
print("[2/7] 加载模型 (float32 MKL)...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    dtype=torch.float32, device_map="cpu",
)
print(f"  模型加载: {time.time()-t0:.0f}s, params={model.num_parameters():,}", flush=True)

# ── LoRA ──
print("[3/7] 应用 LoRA...", flush=True)
lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 数据 ──
print("[4/7] 加载数据...", flush=True)
all_texts = []
with open(TRAIN_DATA, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        record = json.loads(line)
        msgs = [
            {"role": "system", "content": "你是一个在三易约束框架下运行的认知分析模型。"},
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record["output"]},
        ]
        all_texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))

print(f"  样本: {len(all_texts)} 条", flush=True)

tok = tokenizer(all_texts, truncation=True, max_length=MAX_LEN, padding=False)
data = [(torch.tensor(i, dtype=torch.long), torch.tensor(m, dtype=torch.long))
        for i, m in zip(tok["input_ids"], tok["attention_mask"])]

class SimpleDataset(Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, idx):
        i, m = self.d[idx]
        return {"input_ids": i, "attention_mask": m, "labels": i.clone()}

def collate_fn(batch):
    pad_id = tokenizer.pad_token_id
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id),
        "attention_mask": torch.nn.utils.rnn.pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
        "labels": torch.nn.utils.rnn.pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100),
    }

dl = DataLoader(SimpleDataset(data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

num_batches = len(dl)
total_steps = num_batches * NUM_EPOCHS
print(f"[5/7] 训练: {num_batches} batches/epoch x {NUM_EPOCHS} epochs = {total_steps} steps", flush=True)

# ── 训练循环 ──
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    epoch_t0 = time.time()

    for step, batch in enumerate(dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        t_batch = time.time()

        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        batch_time = time.time() - t_batch
        tokens_per_sec = batch["input_ids"].numel() / batch_time

        if (step + 1) % 3 == 0 or step == num_batches - 1:
            tokens_in_batch = batch["input_ids"].numel()
            print(f"  E{epoch+1} B{step+1}/{num_batches} "
                  f"loss={loss.item():.4f} avg_loss={epoch_loss/(step+1):.4f} "
                  f"time={batch_time:.1f}s tok/s={tokens_per_sec:.0f}", flush=True)

    epoch_time = time.time() - epoch_t0
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: avg_loss={avg_loss:.4f}, time={epoch_time:.0f}s", flush=True)

    ep_dir = f"{OUTPUT_DIR}/epoch_{epoch+1}"
    os.makedirs(ep_dir, exist_ok=True)
    model.save_pretrained(ep_dir)
    tokenizer.save_pretrained(ep_dir)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n[6/7] 完成！保存到 {OUTPUT_DIR}", flush=True)
print(f"[7/7] 总耗时: {time.time()-t0:.0f}s", flush=True)
