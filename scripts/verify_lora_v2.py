"""
路线B 验证脚本 — 归藏算法核心验证

归藏引擎的核心逻辑：同一个问题，经历 生→动→长→育 四阶段循环约束。
不是四个不同问题各取一个阶段——那是错的。

测试设计：
1. 选取一个核心问题，分别施加四阶段约束提示
2. 每个阶段对比：基础模型 vs LoRA 模型的约束合规度
3. 重点判断：LoRA 是否比基础模型更严格遵守阶段约束
4. 附加测试：无约束普通对话，检测灾难性遗忘
"""
import torch
import json
import os
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ════════════════════ 配置 ════════════════════
MODEL_PATH = r"C:/Users/VINGO/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct"
LORA_PATH = r"C:/xing/lora_output_v2/best"
OUTPUT_FILE = Path(r"C:/xing/lora_output_v2/verification_report.json")

GEN_CONFIG = dict(max_new_tokens=200, temperature=0.7, do_sample=True,
                  top_p=0.9, repetition_penalty=1.1)

# ════════════════════ 测试用例 ════════════════════
#
# 核心原则：归藏算法是对【同一个问题】施加 生→动→长→育 四阶段循环约束。
# 不是四个不同问题各取一个阶段——那违背了归藏引擎的基本逻辑。

CORE_QUESTION = "技术进步的终点是解放人类还是异化人类？"

CONSTRAINT_SYSTEM = """你是一个运行在三易约束框架中的认知分析模型。

【归藏引擎 — 八气算子】
你的输出按以下四个生成算子循环推进：
- 生（起念探索）：使用试探词（也许/可能/如何/是否），开放提问
- 动（发散联想）：使用发散词（此外/另一方面/还可以/换个角度看），多角度扩展
- 长（收敛聚焦）：使用聚焦词（重点/沿着/深入/核心在于），选一条路径深入
- 育（方案分解）：使用编号结构（第一步/第二步/1./2.），输出可执行子任务

每个算子阶段必须体现对应的思维特征和词语模式。"""

TEST_CASES = [
    # ── 同一个问题，四个阶段循环 ──
    {
        "id": "T1-生",
        "name": f"生阶段 — 对「{CORE_QUESTION[:12]}…」起念探索",
        "system": CONSTRAINT_SYSTEM,
        "user": f"""当前归藏算子：生（起念探索）
周易姿态：兑（表达）

问题：{CORE_QUESTION}

请依照「生」阶段的约束输出：提出开放性问题或假设。使用试探词（也许/可能/如何/是否）。不要给出确定结论，只做探索性思考。
禁止使用：「第一步」「因此」「最终结论」「应该」。""",
        "expect": {
            "must_contain": ["可能", "如何", "是否", "也许"],
            "must_not_contain": ["第一步", "因此", "最终结论", "应该"],
            "style": "生—起念探索：试探、开放、提问"
        }
    },
    {
        "id": "T2-动",
        "name": f"动阶段 — 对「{CORE_QUESTION[:12]}…」发散联想",
        "system": CONSTRAINT_SYSTEM,
        "user": f"""当前归藏算子：动（发散联想）
周易姿态：坎（破局）

问题：{CORE_QUESTION}

请依照「动」阶段的约束输出：从多个角度扩展思考。使用发散词（此外/另一方面/还可以/换个角度看）。列举多种可能性，不急于收敛。
禁止使用：「最终」「应该」「正确方案是」。""",
        "expect": {
            "must_contain": ["此外", "另一方面", "还可以", "换个角度看"],
            "must_not_contain": ["最终", "应该", "正确方案是", "第一步"],
            "style": "动—发散联想：多角度、扩展、不收敛"
        }
    },
    {
        "id": "T3-长",
        "name": f"长阶段 — 对「{CORE_QUESTION[:12]}…」收敛聚焦",
        "system": CONSTRAINT_SYSTEM,
        "user": f"""当前归藏算子：长（收敛聚焦）
周易姿态：震（行动）

问题：{CORE_QUESTION}

请依照「长」阶段的约束输出：从前面发散中选一条路径深入。使用聚焦词（重点/沿着/深入/核心在于）。给出明确的分析方向。
禁止使用：「也许」「可能」「此外」「换个角度看」。必须聚焦，不能发散。""",
        "expect": {
            "must_contain": ["重点", "沿着", "深入", "核心"],
            "must_not_contain": ["也许", "可能", "此外", "换个角度看"],
            "style": "长—收敛聚焦：深入、聚焦、选定路径"
        }
    },
    {
        "id": "T4-育",
        "name": f"育阶段 — 对「{CORE_QUESTION[:12]}…」方案分解",
        "system": CONSTRAINT_SYSTEM,
        "user": f"""当前归藏算子：育（方案分解）
周易姿态：乾（创造）

问题：{CORE_QUESTION}

请依照「育」阶段的约束输出：使用编号结构（第一步/第二步/1./2.）。每个子任务要具体、有操作步骤。
禁止使用：试探词（也许/可能/如何/是否）和发散词（此外/还可以）。必须给出可执行的步骤。""",
        "expect": {
            "must_contain": ["第一步", "第二步", "1.", "2."],
            "must_not_contain": ["也许", "可能", "如何", "是否"],
            "style": "育—方案分解：结构化、编号、可执行"
        }
    },
    # ── 无约束对照组 ──
    {
        "id": "T5-普通",
        "name": "无约束 — 普通知识问答",
        "system": "你是一个有用的AI助手。",
        "user": "什么是项目管理中的资源约束？请简要解释。",
        "expect": {
            "must_contain": [],
            "must_not_contain": [],
            "style": "正常对话"
        }
    },
    {
        "id": "T6-编程",
        "name": "无约束 — 编程问答",
        "system": "你是一个有用的AI助手。",
        "user": "用 Python 写一个函数，判断一个字符串是否是回文。",
        "expect": {
            "must_contain": [],
            "must_not_contain": [],
            "style": "正常编程回答"
        }
    },
]


# ════════════════════ 分析函数 ════════════════════

def analyze_response(response: str, expect: dict) -> dict:
    """分析回复是否满足约束要求"""
    hits = []
    misses = []

    for word in expect.get("must_contain", []):
        if word in response:
            hits.append(word)
        else:
            misses.append(word)

    violations = []
    for word in expect.get("must_not_contain", []):
        if word in response:
            violations.append(word)

    hit_rate = len(hits) / len(expect["must_contain"]) if expect["must_contain"] else 1.0
    violation_rate = len(violations) / len(expect["must_not_contain"]) if expect["must_not_contain"] else 0.0

    return {
        "hits": hits,
        "misses": misses,
        "violations": violations,
        "hit_rate": round(hit_rate, 3),
        "violation_rate": round(violation_rate, 3),
        "passed": hit_rate >= 0.5 and violation_rate == 0,
        "length": len(response),
        "response_preview": response[:200],
    }


def run_inference(model, tokenizer, system: str, user: str) -> str:
    """执行推理"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)

    with torch.no_grad():
        outputs = model.generate(**inputs, **GEN_CONFIG)

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return response.strip()


# ════════════════════ 主流程 ════════════════════

def main():
    print("=" * 70)
    print("  象·路线B 约束内化验证")
    print("=" * 70)

    # ── 加载模型 ──
    print(f"\n[1/3] 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"  ✓ 基础模型加载完成")

    # ── 加载LoRA ──
    print(f"\n[2/3] 加载 LoRA 权重: {LORA_PATH}")
    adapter_file = Path(LORA_PATH) / "adapter_model.safetensors"
    adapter_bin = Path(LORA_PATH) / "adapter_model.bin"

    has_lora = False
    lora_model = None

    if adapter_file.exists() or adapter_bin.exists():
        try:
            lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)
            lora_model.eval()
            has_lora = True
            print(f"  ✓ LoRA 加载成功！")
            if adapter_file.exists():
                print(f"    文件: adapter_model.safetensors ({adapter_file.stat().st_size/1024:.0f} KB)")
            else:
                print(f"    文件: adapter_model.bin ({adapter_bin.stat().st_size/1024:.0f} KB)")
        except Exception as e:
            print(f"  ✗ LoRA 加载失败: {e}")
    else:
        print(f"  ✗ LoRA 文件不存在！")
        print(f"    检查路径: {LORA_PATH}")
        found_files = list(Path(LORA_PATH).glob("*")) if Path(LORA_PATH).exists() else []
        print(f"    目录内容: {[f.name for f in found_files]}")
        return

    # ── 运行测试 ──
    print(f"\n[3/3] 运行验证测试 ({len(TEST_CASES)} 个用例)...")

    base_model.eval()

    results = {
        "model": "Qwen2.5-0.5B-Instruct",
        "lora_path": str(LORA_PATH),
        "lora_config": "r=16, alpha=32, target=q/k/v/o+gate/up/down",
        "tests": [],
        "summary": {},
    }

    for tc in TEST_CASES:
        print(f"\n{'─' * 70}")
        print(f"  {tc['id']}: {tc['name']}")
        print(f"  预期风格: {tc['expect']['style']}")
        print(f"  必须包含: {tc['expect']['must_contain']}")
        print(f"  禁止包含: {tc['expect']['must_not_contain']}")

        # 基础模型（无LoRA）
        base_resp = run_inference(base_model, tokenizer, tc["system"], tc["user"])
        base_analysis = analyze_response(base_resp, tc["expect"])

        # LoRA模型
        lora_resp = ""
        lora_analysis = {}
        if has_lora:
            lora_resp = run_inference(lora_model, tokenizer, tc["system"], tc["user"])
            lora_analysis = analyze_response(lora_resp, tc["expect"])

        # 打印对比
        print(f"\n  ┌─ 基础模型回复 ─────────────────────────────")
        for line in base_resp[:300].split("\n"):
            print(f"  │ {line}")
        print(f"  └── 命中:{base_analysis['hits']} 遗漏:{base_analysis['misses']} 违规:{base_analysis['violations']}")
        print(f"     通过: {'✅' if base_analysis['passed'] else '❌'}")

        if has_lora:
            print(f"\n  ┌─ LoRA 模型回复 ─────────────────────────────")
            for line in lora_resp[:300].split("\n"):
                print(f"  │ {line}")
            print(f"  └── 命中:{lora_analysis['hits']} 遗漏:{lora_analysis['misses']} 违规:{lora_analysis['violations']}")
            print(f"     通过: {'✅' if lora_analysis['passed'] else '❌'}")

        results["tests"].append({
            "id": tc["id"],
            "name": tc["name"],
            "expect": tc["expect"],
            "base": {
                "response": base_resp,
                "analysis": base_analysis,
            },
            "lora": {
                "response": lora_resp,
                "analysis": lora_analysis,
            } if has_lora else None,
        })

    # ── 统计总结 ──
    print(f"\n{'=' * 70}")
    print(f"  验证总结")
    print(f"{'=' * 70}")

    constrained_tests = [t for t in results["tests"] if t["id"].startswith("T1") or t["id"].startswith("T2") or t["id"].startswith("T3") or t["id"].startswith("T4")]
    unconstrained_tests = [t for t in results["tests"] if t["id"].startswith("T5") or t["id"].startswith("T6")]

    # 约束测试
    base_constrained_pass = sum(1 for t in constrained_tests if t["base"]["analysis"]["passed"])
    lora_constrained_pass = sum(1 for t in constrained_tests if t["lora"] and t["lora"]["analysis"]["passed"])

    print(f"\n  约束内化测试 (T1-T4):")
    print(f"    基础模型: {base_constrained_pass}/4 通过")
    if has_lora:
        print(f"    LoRA模型: {lora_constrained_pass}/4 通过")
        improvement = lora_constrained_pass - base_constrained_pass
        print(f"    提升: {'+' if improvement >= 0 else ''}{improvement} 项")

    # 普通能力保持
    base_normal = all(t["base"]["analysis"].get("length", 0) > 20 for t in unconstrained_tests)
    lora_normal = all(t["lora"] and t["lora"]["analysis"].get("length", 0) > 20 for t in unconstrained_tests) if has_lora else False
    print(f"\n  普通能力保持 (T5-T6):")
    print(f"    基础模型: {'✅ 正常' if base_normal else '❌ 异常'}")
    if has_lora:
        print(f"    LoRA模型: {'✅ 正常' if lora_normal else '❌ 异常 (灾难性遗忘!)'}")

    # 综合判断
    results["summary"] = {
        "base_constrained_pass": base_constrained_pass,
        "lora_constrained_pass": lora_constrained_pass,
        "base_normal": base_normal,
        "lora_normal": lora_normal,
        "recommendation": "",
    }

    if has_lora:
        if lora_constrained_pass >= 3 and lora_normal:
            results["summary"]["recommendation"] = "✅ 路线B约束内化生效！建议购入5060 Ti 16GB推进14B模型训练。"
        elif lora_constrained_pass >= 2:
            results["summary"]["recommendation"] = "⚠️ 约束内化有部分效果但不稳定。建议：1) 增加训练数据到500条 2) 增加epochs到8 3) 在0.5B确认稳定后再推进GPU。"
        else:
            results["summary"]["recommendation"] = "❌ 约束内化效果不足。建议：1) 检查训练数据质量 2) 增加LoRA rank到32 3) 切换更大的base模型（7B）再试。暂时暂缓购入GPU。"
    else:
        results["summary"]["recommendation"] = "❌ LoRA权重未加载成功，无法评估。"

    print(f"\n  📋 综合建议: {results['summary']['recommendation']}")

    # 保存报告
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  详细报告已保存: {OUTPUT_FILE}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
