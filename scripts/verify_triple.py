"""
路线B 三组对照验证 — llama-server HTTP API + Vulkan GPU
使用 llama-server REST API 进行推理，避免 llama-cli 交互模式问题。

三组对照：
  A. 约束组：合并模型 + 动态阶段提示（无体系框架）
  B. 微调组：合并模型 + 裸问题
  C. 原生组：基础模型 + 裸问题
"""
import subprocess
import json
import time
import urllib.request
import sys
from pathlib import Path

# ════════════════════ 配置 ════════════════════
LLAMA_SERVER = r"C:/xing/llama.cpp/build/bin/Release/llama-server.exe"
MERGED_MODEL = r"C:/xing/models/xiang-routeb-0.5b-merged.gguf"
BASE_GGUF = r"C:/xing/models/qwen2.5-0.5b-instruct-f16.gguf"
OUTPUT_FILE = Path(r"C:/xing/lora_output_v2/triple_verification.json")

SERVER_PORT = 8087
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

GPU_LAYERS = 99
MAX_TOKENS = 256
THREADS = 8
TEMPERATURE = 0.7

CORE_QUESTION = "技术进步的终点是解放人类还是异化人类？"

PHASE_PROMPTS = {
    "生": {
        "name": "生—起念探索",
        "user": (
            f"当前阶段：起念探索。\n"
            f"任务：{CORE_QUESTION}\n"
            f"要求：使用试探性语气，提出开放性问题或假设。"
            f"使用词语如 也许、可能、如何、是否。不要下结论。"
        ),
        "keywords": ["可能", "如何", "是否", "也许"],
        "forbidden": ["第一步", "因此", "最终结论", "应该"],
    },
    "动": {
        "name": "动—发散联想",
        "user": (
            f"当前阶段：发散联想。\n"
            f"任务：{CORE_QUESTION}\n"
            f"要求：从多个角度扩展思考。"
            f"使用词语如 此外、另一方面、还可以、换个角度看。不要收敛。"
        ),
        "keywords": ["此外", "另一方面", "还可以", "换个角度看"],
        "forbidden": ["最终", "应该", "正确方案是"],
    },
    "长": {
        "name": "长—收敛聚焦",
        "user": (
            f"当前阶段：收敛聚焦。\n"
            f"任务：{CORE_QUESTION}\n"
            f"要求：选定一条路径深入分析。"
            f"使用词语如 重点、沿着、深入、核心在于。必须聚焦，不能发散。"
        ),
        "keywords": ["重点", "沿着", "深入", "核心"],
        "forbidden": ["也许", "可能", "此外", "换个角度看"],
    },
    "育": {
        "name": "育—方案分解",
        "user": (
            f"当前阶段：方案分解。\n"
            f"任务：{CORE_QUESTION}\n"
            f"要求：输出结构化步骤。使用编号（第一步、第二步、1.、2.）。"
            f"每个步骤要具体、可执行。禁止试探和发散。"
        ),
        "keywords": ["第一步", "第二步", "1.", "2."],
        "forbidden": ["也许", "可能", "如何", "是否"],
    },
}


# ════════════════════ HTTP API 推理 ════════════════════

def chat_completion(system: str, user: str, max_tokens: int = 256) -> str:
    """调用 llama-server /v1/chat/completions"""
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        content = result["choices"][0]["message"]["content"].strip()
        print(f"    [{elapsed:.1f}s] {len(content)} chars")
        return content
    except Exception as e:
        print(f"    [错误] {e}")
        return f"[ERROR: {e}]"


def analyze(phase: str, response: str) -> dict:
    """分析回复是否符合该阶段约束"""
    spec = PHASE_PROMPTS[phase]
    hits = [w for w in spec["keywords"] if w in response]
    violations = [w for w in spec["forbidden"] if w in response]
    hit_rate = len(hits) / len(spec["keywords"])
    violation_rate = len(violations) / len(spec["forbidden"]) if spec["forbidden"] else 0

    return {
        "hits": hits,
        "hit_rate": round(hit_rate, 2),
        "violations": violations,
        "violation_rate": round(violation_rate, 2),
        "passed": hit_rate >= 0.5 and violation_rate == 0,
        "length": len(response),
    }


# ════════════════════ 服务管理 ════════════════════

def start_server(model_path: str) -> subprocess.Popen:
    """启动 llama-server"""
    cmd = [
        LLAMA_SERVER,
        "-m", model_path,
        "--port", str(SERVER_PORT),
        "-ngl", str(GPU_LAYERS),
        "-t", str(THREADS),
        "--host", "127.0.0.1",
        "--no-webui",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # 等服务器就绪
    print(f"  启动 llama-server (port {SERVER_PORT})...")
    for i in range(30):
        time.sleep(1)
        try:
            req = urllib.request.Request(f"{SERVER_URL}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(f"  ✓ 服务就绪")
                    return proc
        except Exception:
            pass
    raise RuntimeError("llama-server 启动超时")


def stop_server(proc: subprocess.Popen):
    """停止 llama-server"""
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


# ════════════════════ 主流程 ════════════════════

def main():
    print("=" * 70)
    print("  路线B 三组对照验证 — Vulkan GPU (HTTP API)")
    print("=" * 70)

    if not Path(MERGED_MODEL).exists():
        print(f"\n❌ 合并模型不存在: {MERGED_MODEL}")
        sys.exit(1)
    if not Path(BASE_GGUF).exists():
        print(f"\n❌ 基础模型不存在: {BASE_GGUF}")
        sys.exit(1)

    results = {
        "question": CORE_QUESTION,
        "groups": {},
        "summary": {},
    }

    # ── A组 & B组：使用合并模型 ──
    print(f"\n[模型1] 合并LoRA模型: {Path(MERGED_MODEL).name}")
    server_proc = start_server(MERGED_MODEL)

    for group_id, group_label, use_constraint in [
        ("A-约束组", "LoRA合并 + 动态阶段提示", True),
        ("B-微调组", "LoRA合并 + 裸问题", False),
    ]:
        print(f"\n{'─' * 70}")
        print(f"  【{group_id}】{group_label}")
        print(f"{'─' * 70}")

        group_result = {}
        for phase, spec in PHASE_PROMPTS.items():
            print(f"\n  ▶ {spec['name']}")

            if use_constraint:
                user_prompt = spec["user"]
            else:
                user_prompt = f"问题：{CORE_QUESTION}\n请回答。"

            response = chat_completion("", user_prompt)
            analysis = analyze(phase, response)

            group_result[phase] = {
                "response": response,
                "analysis": analysis,
                "passed": analysis["passed"],
            }

            status = "✅" if analysis["passed"] else "❌"
            print(f"    命中:{analysis['hits']} 违规:{analysis['violations']} {status}")
            # 打印实际生成内容
            for line in response[:400].split('\n'):
                print(f"    │ {line[:100]}")
            print(f"    └── {len(response)} chars")

        results["groups"][group_id] = group_result

        passed_count = sum(1 for v in group_result.values() if v["passed"])
        print(f"\n  [{group_id}] 阶段通过: {passed_count}/4")

    stop_server(server_proc)

    # ── C组：使用基础模型 ──
    print(f"\n\n[模型2] 基础模型: {Path(BASE_GGUF).name}")
    server_proc = start_server(BASE_GGUF)

    group_id = "C-原生组"
    print(f"\n{'─' * 70}")
    print(f"  【{group_id}】基础模型 + 裸问题")
    print(f"{'─' * 70}")

    group_result = {}
    for phase, spec in PHASE_PROMPTS.items():
        print(f"\n  ▶ {spec['name']}")
        user_prompt = f"问题：{CORE_QUESTION}\n请回答。"

        response = chat_completion("", user_prompt)
        analysis = analyze(phase, response)

        group_result[phase] = {
            "response": response,
            "analysis": analysis,
            "passed": analysis["passed"],
        }

        status = "✅" if analysis["passed"] else "❌"
        print(f"    命中:{analysis['hits']} 违规:{analysis['violations']} {status}")
        # 打印实际生成内容
        for line in response[:400].split('\n'):
            print(f"    │ {line[:100]}")
        print(f"    └── {len(response)} chars")

    stop_server(server_proc)
    results["groups"][group_id] = group_result

    passed_count = sum(1 for v in group_result.values() if v["passed"])
    print(f"\n  [{group_id}] 阶段通过: {passed_count}/4")

    # ── 对比分析 ──
    print(f"\n{'=' * 70}")
    print(f"  三组差距分析")
    print(f"{'=' * 70}")

    summary = {}
    for gid in ["A-约束组", "B-微调组", "C-原生组"]:
        grp = results["groups"][gid]
        passed = sum(1 for v in grp.values() if v["passed"])
        avg_hit = sum(v["analysis"]["hit_rate"] for v in grp.values()) / 4
        avg_viol = sum(v["analysis"]["violation_rate"] for v in grp.values()) / 4
        summary[gid] = {"passed": passed, "avg_hit": round(avg_hit, 2),
                         "avg_viol": round(avg_viol, 2)}

        print(f"\n  {gid}: {passed}/4 通过, "
              f"命中率 avg={avg_hit:.2f}, 违规率 avg={avg_viol:.2f}")

    delta_constraint = summary["A-约束组"]["passed"] - summary["B-微调组"]["passed"]
    delta_lora = summary["B-微调组"]["passed"] - summary["C-原生组"]["passed"]
    print(f"\n  差距:")
    print(f"    动态约束加成 = {delta_constraint:+d} (约束组 - 微调组)")
    print(f"    LoRA 效应     = {delta_lora:+d} (微调组 - 原生组)")

    results["summary"] = summary
    results["summary"]["delta_constraint"] = delta_constraint
    results["summary"]["delta_lora"] = delta_lora

    if delta_constraint > 0 and delta_lora > 0:
        results["summary"]["conclusion"] = "✅ 双效验证：LoRA改变了原生行为，动态约束继续有效。"
    elif delta_lora > 0:
        results["summary"]["conclusion"] = "⚠️ LoRA改变了原生行为但动态约束无效。"
    elif delta_constraint > 0:
        results["summary"]["conclusion"] = "⚠️ 动态约束有效但LoRA未改变原生行为。0.5B太小，需换7B。"
    else:
        results["summary"]["conclusion"] = "❌ 两者无显著效应。0.5B太小，需在7B上重新评估。"

    print(f"\n  📋 结论: {results['summary']['conclusion']}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  报告: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
