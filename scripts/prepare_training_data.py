"""
藏海·约束理解微调 — 训练数据生成 v3
用最可靠的方式从基准测试轮次文件中提取数据。
"""

import os, re, json, glob

ROOT = r"C:/xing/benchmark_output"
OUT = r"C:/xing/training_data.jsonl"

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_turn(text):
    """将轮次文件解析为字段dict——用split分割，最可靠"""
    # 按"── fieldname ──" 分割
    parts = re.split(r'\n──\s*(.+?)\s*──\n', text)
    result = {}
    # parts[0] = 文件头（空或无关内容）
    # parts[1] = 第一个字段名, parts[2] = 第一个字段内容
    # parts[3] = 第二个字段名, parts[4] = 第二个字段内容...
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].strip()
        val = parts[i + 1].strip()
        result[key] = val
    return result

# ── 扫描所有trial ──
trials = sorted(glob.glob(os.path.join(ROOT, "trial_*")))
all_data = []

for tp in trials:
    tid = os.path.basename(tp)
    exp_dir = os.path.join(tp, "实验组")
    ctrl_dir = os.path.join(tp, "对照组")
    
    if not os.path.isdir(exp_dir) or not os.path.isdir(ctrl_dir):
        continue
    
    exp_files = sorted(glob.glob(os.path.join(exp_dir, "turn_*.txt")))
    ctrl_files = sorted(glob.glob(os.path.join(ctrl_dir, "turn_*.txt")))
    
    if not exp_files or not ctrl_files:
        continue
    
    # 解析所有轮次
    exp_turns = {}
    for f in exp_files:
        m = re.search(r'turn_(\d+)', os.path.basename(f))
        if m:
            step = int(m.group(1))
            exp_turns[step] = parse_turn(read_file(f))
    
    ctrl_turns = {}
    for f in ctrl_files:
        m = re.search(r'turn_(\d+)', os.path.basename(f))
        if m:
            step = int(m.group(1))
            ctrl_turns[step] = parse_turn(read_file(f))
    
    # 提取约束系统提示词
    system_prompt = ""
    for step in sorted(exp_turns.keys()):
        sp = exp_turns[step].get("系统提示词", "")
        if sp:
            system_prompt = sp
            break
    if not system_prompt:
        print(f"  {tid}: 跳过 - 无系统提示词")
        continue
    
    # 判断实验组哪些轮次是OK的
    ok_steps = set()
    for f in exp_files:
        m = re.search(r'turn_(\d+)_(.+?)\.', os.path.basename(f))
        if m:
            step = int(m.group(1))
            label = m.group(2)
            if "_OK" in f or label == "OK":
                ok_steps.add(step)
    
    # 构建样本
    all_steps = sorted(set(list(exp_turns.keys()) + list(ctrl_turns.keys())))
    
    for step in all_steps:
        user_input = ""
        if step in exp_turns:
            user_input = exp_turns[step].get("用户输入", "")
        if not user_input and step in ctrl_turns:
            user_input = ctrl_turns[step].get("用户输入", "")
        if not user_input:
            continue
        
        ctrl_output = ctrl_turns[step].get("生成输出（全文）", "") if step in ctrl_turns else ""
        exp_output = exp_turns[step].get("生成输出（全文）", "") if step in exp_turns else ""
        
        # 样本A：约束提示 + 对照组输出
        if ctrl_output and len(ctrl_output) >= 100:
            inst = f"""你运行在三易约束框架中。系统通过以下引擎引导你的分析：

{system_prompt[:2000]}

当前轮次：
{user_input[:2000]}

在三易约束框架的引导下完成本轮分析。"""
            all_data.append({
                "instruction": inst.strip(),
                "output": ctrl_output,
                "source": f"{tid}/对照/{step}",
                "type": "ctrl",
            })
        
        # 样本B：约束提示 + 实验组OK输出
        if exp_output and step in ok_steps and len(exp_output) >= 50:
            inst = f"""你运行在三易约束框架中。系统通过以下引擎引导你的分析：

{system_prompt[:2000]}

当前轮次：
{user_input[:2000]}

在三易约束框架的引导下完成本轮分析。"""
            all_data.append({
                "instruction": inst.strip(),
                "output": exp_output,
                "source": f"{tid}/实验/{step}",
                "type": "exp",
            })
    
    count = len([d for d in all_data if d['source'].startswith(tid)])
    print(f"  {tid}: {count} 条 (exp_ok={len([d for d in all_data if d['source'].startswith(tid) and d['type']=='exp'])})")

# ── 去重 ──
seen = set()
deduped = []
for item in all_data:
    key = item["output"][:100]
    if key not in seen:
        seen.add(key)
        deduped.append(item)

# ── 写出 ──
with open(OUT, 'w', encoding='utf-8') as f:
    for item in deduped:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

types = {}
for item in deduped:
    types[item["type"]] = types.get(item["type"], 0) + 1

print(f"\n总计: {len(all_data)} 条 → 去重后 {len(deduped)} 条")
print(f"类型: {types}")

if deduped:
    s = deduped[0]
    print(f"\n=== 样例 ===")
    print(f"来源: {s['source']}, 类型: {s['type']}")
    print(f"指令({len(s['instruction'])}字): {s['instruction'][:200]}...")
    print(f"输出({len(s['output'])}字): {s['output'][:200]}...")
