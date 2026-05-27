# Xiang (象) — Three-Yi Intelligent Constraint System

> **Not to make the model smarter, but to give it structured awareness calibration, strategy selection, and posture adjustment capabilities.**

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2024-edition?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Status](https://img.shields.io/badge/Status-Alpha-blue)](ARCHITECTURE.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](ARCHITECTURE.md)

</div>

---

## One-Sentence Definition

**Xiang (象)** is a inference-time cognitive constraint framework based on **XiangLang (象语言)**, unifying three independent Yi-Jing operator systems (GuiZang / LianShan / ZhouYi) into a single constraint layer for programmatic control over LLM generation.

This is NOT a reasoning enhancement tool, NOT a context manager, and NOT prompt engineering. This is something that doesn't exist on the market today — a **small-model Agent framework with rudimentary metacognitive capabilities that can self-evolve**.

---

## Project Structure: Three Engines, One Language

```
                    ┌──────────────────────────────┐
                    │   LianShan (ShanVM)           │
                    │   Strategy Engine · Navigation │
                    │                              │
                    │   What to do when stuck?     │
                    │   7 Azimuth Decisions:        │
                    │   Gate→Qi→Jia→Yuan→Zhi→Jue  │
                    └──────────┬───────────────────┘
                              │ "Strategy advice"
                              ▼
  ┌─────────────────────────────────────────────────┐
  │           GuiZang (CangVM)                       │
  │       Execution Controller · 8-Operator Cycle    │
  │                                                   │
  │  Sheng→Dong→Gui→Zhang→Yu→Sha→Zhi→Zang           │
  │  Decisions: PASS / ROLLBACK / SKIP / STOP         │
  │                                                   │
  │  Receives deviation signals, strategy advice,     │
  │  cognitive posture, and makes execution decisions │
  └──┬──────────────┬──────────────────┬─────────────┘
     │              │                  │
     │ "Deviation"  │ "Posture"        │ "Experience"
     ▼              ▼                  ▼
  ┌──────────┐  ┌──────────┐    ┌──────────┐
  │ YinYi   │  │ ZhouYi  │    │ CangSea │
  │ Regex   │  │(ZhouVM) │    │Hebbian  │
  │ Checker │  │Posture  │    │ Memory  │
  │ Format  │  │Bagua    │    │ Semantic│
  │ Valid.  │  │Grid     │    │ Weights │
  └──────────┘  └──────────┘    └──────────┘
```

### Engine 1: GuiZang (CangVM) — Execution Controller

**Implemented and fully integrated with real LLM.**

| Operator | Type | Purpose | Status |
|:----|:----|------|:--------:|
| **Sheng** | Generate | Initiate exploration | ✅ |
| **Dong** | Generate | Divergent association | ✅ |
| **Zhang** | Generate | Direction clarification | ✅ |
| **Yu** | Generate | Plan decomposition | ✅ |
| **Gui** | Control | Deviation detection | ✅ |
| **Sha** | Control | Truncation & rollback | ✅ |
| **Zhi** | Control | Cycle termination | ✅ |
| **Zang** | Control | Experience sedimentation | ✅ |

The core decision logic `judge()` returns 5 instructions: `PASS` / `ROLLBACK` / `SKIP` / `STOP` / `FINISH_CYCLE`.

### Engine 2: LianShan (ShanVM) — Strategy Navigation

**Code implemented, not yet connected to main pipeline.**

When the model encounters obstacles (persistently high deviation, multiple retries), LianShan takes over:

| Step | Name | Check |
|:---:|:----|---------|
| 1 | **Gate** | Activation threshold reached? |
| 2 | **Qi** | Which phase: Spring/Summer/Autumn/Winter |
| 3 | **Jia** | Obstacle level: Initial / Secondary / Tertiary |
| 4 | **Yuan** | Context freshness: New / Stale |
| 5 | **Zhi** | Push vs. Resistance evaluation |
| 6 | **Jue** | Output one of 7 azimuth decisions |

| Decision | Meaning |
|----------|------|
| **Continue** | Continue current path |
| **PushThrough** | Force through — increase bias intensity |
| **NavigateAround** | Detour — change problem approach angle |
| **Decompose** | Break current problem into subtasks |
| **Escalate** | Request external help or switch strategy |
| **Abort** | Terminate current task |
| **WaitGather** | Gather more context before deciding |

### Engine 3: ZhouYi (ZhouVM) — Cognitive Posture

**Code implemented, not yet connected to main pipeline.**

ZhouYi determines what **cognitive posture** the model uses to handle the current task. It controls temperature and behavioral tendencies through an 8-trigram grid + Five Elements interaction weight matrix:

| Trigram | Element | Posture | Temp | Use Case |
|:---:|:---:|:-----|:---:|---------|
| **Qian** | Metal | Create | 1.2 | Brainstorming, creative generation |
| **Dui** | Metal | Express | 0.9 | Communication, teaching |
| **Li** | Fire | Illuminate | 0.5 | Deep analysis, focused review |
| **Zhen** | Wood | Initiate | 1.0 | New task initialization |
| **Xun** | Wind | Permeate | 0.7 | Progressive reasoning |
| **Kan** | Water | Breakthrough | 1.1 | Thinking outside the box |
| **Gen** | Mountain | Stabilize | 0.3 | Precision focus, strict convergence |
| **Kun** | Earth | Sustain | 0.6 | Default stable state |

Grid state transitions follow Five Elements interaction rules, driven by an 8×8 Markov weight matrix.

---

## XiangLang: Unified Constraint Language for Three Operator Systems

**XiangLang is NOT syntactic sugar for GuiZang. It is the unified constraint language for three independent operator systems.**

All three engines share the same underlying type system (defined in `xiang-core`):

| Shared Type | Purpose | Used By |
|---------|------|--------------|
| **Gua(u8)** | 6-bit trigram state (0-63) | All three engines |
| **Eight Operators** | Sheng/Dong/Gui/Zhang/Yu/Sha/Zhi/Zang | CangVM executes, ShanVM outputs, ZhouVM adjusts |
| **FangWei** | 7 azimuth decisions | ShanVM produces, CangVM consumes |
| **Bagua** | 8 trigrams + interaction weight matrix | ZhouVM core, CangVM reads temperature |
| **CangSea** | 64×64 Hebbian experience matrix + semantic storage | GuiZang sediments, LianShan/ZhouYi consume |

---

## Route Evolution

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Now: Route A                     Future: Route B→C          │
│  ┌───────────────────┐         ┌──────────────────────┐     │
│  │ Three-Layer Const │   →    │ Unified Three-Yi      │     │
│  │ (Complete)        │         │ (Full Synergy)       │     │
│  │                   │         │                      │     │
│  │ · CangVM State    │         │ · CangVM + ShanVM    │     │
│  │ · YinProtocol     │         │   + ZhouVM 3-Layer   │     │
│  │ · Logit-Bias      │         │ · CangSea + LoRA     │     │
│  │ · HttpBackend     │         │ · Cloud Judge API    │     │
│  │ · Zero Prompt Inj │         │ · Flywheel Iteration │     │
│  │ · chat-ui         │         │ · Self-Evolution     │     │
│  └───────────────────┘         └──────────────────────┘     │
│                                                             │
│   ShanVM ready ──→ needs pipeline integration               │
│   ZhouVM ready ──→ needs pipeline integration               │
│   CangSea export ──→ needs training data pipeline           │
│   LoRA fine-tune ──→ needs Route B vocab + scripts          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Comparison with Existing Technologies

| System | Strategy Nav? | Cognitive Posture? | Constraint Engine? | Self-Evolution? |
|------|:---:|:---:|:---:|:---:|
| LangGraph / AutoGen | Task orchestration | ❌ | ❌ | ❌ |
| Distillation | ❌ | ❌ | ❌ | One-shot |
| STaR / ReST | ❌ | ❌ | ❌ | Verifiable tasks only |
| **Xiang / Three-Yi** | **LianShan 7-azimuth** | **ZhouYi 8-trigram** | **3-layer programmatic** | **Infinite flywheel** |

---

## Current Implementation Status

| Component | Engine | Status |
|------|:----:|:----:|
| CangVM Core (8 operators + judge()) | GuiZang | ✅ Complete |
| YinProtocolChecker (regex validation) | GuiZang | ✅ Complete |
| XiangLogitBias (continuous scaling) | GuiZang | ✅ Complete |
| LlamaCppBackend / HttpBackend | GuiZang | ✅ Complete |
| ConstrainedEngine (chat pipeline) | GuiZang | ✅ Integrated |
| Benchmark framework | GuiZang | ✅ Complete |
| **ShanVM (7-azimuth + strategy)** | **LianShan** | **✅ Code ready, not pipelined** |
| **ZhouVM (8-trigram + temperature)** | **ZhouYi** | **✅ Code ready, not pipelined** |
| Three-engine coordination (event bus) | Three-Yi | 📋 To design |
| CangSea training data export | GuiZang→B | 📋 To implement |
| LoRA fine-tuning pipeline (Route B) | Route B | 📋 **Hardware bottleneck** |
| Cloud judge API | Three-Yi | 📋 To design |
| chat-ui frontend | Route A | ✅ |
| End-to-end test (Qwen3.5-4B, 64K) | Route A | ✅ Passed |

---

## Hardware Requirements

| Tier | Model | VRAM | Context |
|------|------|------|--------|
| Lightest | Qwen2.5-0.5B | ~1-2 GB | 32K |
| Recommended | Qwen2.5-1.5B | ~2-3 GB | 32K-64K |
| Current test | Qwen3.5-4B | ~5 GB | 64K |

### Hardware Limitation Notice

This project is developed and tested on **AMD RX 6650 XT (8GB VRAM)**. Route A (inference-time external constraints) is verified and connected to a real model. However, Route B (LoRA fine-tuning to internalize constraint primitives) is blocked on current hardware:

- 4B QLoRA fine-tuning requires ~7-8GB VRAM, but **no training toolchain is available for AMD on Windows** (bitsandbytes is CUDA-only)
- Current strategy: accumulate CangSea training data via Route A, execute Route B when training environment is available.
- LianShan/ZhouYi pipeline integration has been decoupled from Route B into independent tasks, not blocked by fine-tuning.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for details.

---

## Quick Start

```bash
# One-command startup (llama-server + xiang-chat + chat-ui)
cd c:/xing && start_all.bat
```

Open `http://localhost:5173` in your browser.

Currently starts the GuiZang engine (Route A). LianShan/ZhouYi integration and flywheel training are next steps.

---

## Documentation

| File | Content |
|------|------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System architecture, dependencies, hardware constraints, roadmap |
| [specs/01-路线A-阶段约束.md](./specs/01-路线A-阶段约束.md) | GuiZang engine specification (Chinese) |
| [specs/02-路线B-控制原语微调.md](./specs/02-路线B-控制原语微调.md) | LoRA fine-tuning + 14 control primitives (Chinese) |
| [specs/03-路线C-太极双LLM.md](./specs/03-路线C-太极双LLM.md) | Dual-LLM + semantic encoding (Chinese) |
| [specs/04-象语言规范.md](./specs/04-象语言规范.md) | XiangLang syntax & three-engine specification (Chinese) |
| [specs/05-三易算法内核.md](./specs/05-三易算法内核.md) | Algorithm definitions & deviation methodology (Chinese) |
| [specs/06-五层认知知识文件系统.md](./specs/06-五层认知知识文件系统.md) | Five-layer cognitive knowledge filesystem (Chinese) |
| [specs/07-硬件适配与异构部署方案.md](./specs/07-硬件适配与异构部署方案.md) | Hardware adaptation & heterogeneous deployment (Chinese) |
| [specs/路线C-实施计划.md](./specs/路线C-实施计划.md) | Route C 9-phase implementation plan (Chinese) |

---

## Open Source & Contributing

### License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE).

### Most Needed Contributions

| Area | Description | Difficulty |
|------|------|:----:|
| **Route B training pipeline** | Build QLoRA fine-tuning scripts (external training environment) | ⭐⭐⭐ |
| **AMD GPU training support** | Resolve training toolchain for AMD ROCm | ⭐⭐⭐⭐ |
| **ShanVM pipeline integration** | Connect LianShan to CangVM main loop | ⭐⭐ |
| **ZhouVM pipeline integration** | Connect ZhouYi to CangVM main loop | ⭐⭐ |
| **CangSea data export** | Export training data as JSONL | ⭐⭐ |
| **New operator rules** | Add regex rules to YinProtocolChecker | ⭐ |
| **Frontend enhancement** | chat-ui interaction & visualization | ⭐ |

### Acknowledgements

This project is developed and tested on **AMD RX 6650 XT (8GB VRAM)**. Hardware limitations exposed the ceiling of Route A external constraints and pointed to the necessity of Route B fine-tuning. Thanks to all followers and contributors.

---

> **Architecture Document**: [ARCHITECTURE.md](./ARCHITECTURE.md) — detailed system architecture, component dependencies, hardware constraints, and roadmap.
>
> **Xiang (象)** — Three-Yi Intelligent Constraint System, based on XiangLang. GuiZang, LianShan, ZhouYi — three Yi-Jing operator systems unified in one framework.
>
> *GuiZang executes, LianShan navigates, ZhouYi postures. Together, they form a complete cognitive constraint closed loop.*
