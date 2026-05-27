# 路线C：太极双LLM — 三易全链路语义化架构

**版本：2.2** | **难度：⭐⭐⭐** | **状态：远期愿景——语义引擎设计完成，含完整工程实施计划**

> ⚠️ **本文档描述的是远期架构愿景，含完整工程实施计划。** 路线A（归藏引擎 — ✅ 已实现）已完成。路线B 和连山/周易接线已解耦为独立任务。路线C 的前置条件是连山/周易流水线接入 + 阴仪编码基础设施，不再依赖路线B 的模型微调。
>
> **在三易统一中的位置**：路线C 不是"更大的 归藏"——它是三引擎（归藏/连山/周易）从"离散查表"跃升为"全语义驱动"的完整异步流水线。
>
> **核心升级**：引入阴仪 TextEncoder → 藏海从 64×64 矩阵升级为语义向量数据库 → 连山/周易从固定规则变为语义插值。

**前置依赖**：
- ✅ 路线A — 归藏引擎实现（外挂约束）
- ✅ 连山/周易 ShanVM/ZhouVM — Rust 代码就绪，可独立接线
- 🔜 连山/周易 流水线接入 — 已从路线B 解耦，可独立执行
- 🔜 阴仪 TextEncoder — 可独立实现（MockEncoder 先行）

---

## 0B 硬件说明

路线C 需要**两份模型推理实例**（阳仪生成 + 阴仪编码），显存需求是路线A/B 的两倍。

| 模型 | 双实例 VRAM | 可行性 |
|------|:---:|:---:|
| 0.5B | ~2GB | 可行但语义编码精度有限 |
| 4B | ~10GB | 推荐（当前测试环境 8GB + 量化） |
| 7B | ~28GB | 需高端硬件 |

> 推荐从 4B 模型开始路线C 验证。4B 在路线A 已验证具备稳定指令跟随能力，其编码质量足以驱动有意义的语义偏离度计算。

---

## 一、三易语义化：从离散到连续

### 1.1 当前的离散模式（路线A）

```
当前三易系统:
  · 偏离度 = hamming(state, origin) / 6  → 仅 7 个离散级别（0/6, 1/6, ..., 6/6）
  · 藏海 = 64×64 Hebbian 矩阵 → 纯离散状态转移
  · 连山 = 6 步查表决策 → 固定规则，不读上下文
  · 周易 = 8 卦 Markov 链 → 固定生克权重 + 温度
```

### 1.2 路线C 的语义模式

引入**阴仪（TextEncoder）**后，整个三易系统从"离散查表"跃升为"语义驱动"：

```
三易语义化系统:
  · 偏离度 = hybrid_deviation = α·hamming + (1-α)·cosine
  · 藏海 = 64×64 矩阵 (legacy) + SemanticStore (向量数据库)
  · 连山 = V_goal + V_obstacle + C_history → V_strategy（语义策略分类器）
  · 周易 = V_origin + V_strategy → V_attitude（认知姿态编码器 → 含温度 + prompt_prefix）
```

**核心突破**：连山和周易不再是无状态规则查表——它们有了自己的语义内存（藏海 SemanticStore），可以记忆过去的成功策略和失败模式。

---

## 二、架构总览

```
                         ┌── 阴仪 (Yin) ──────────────────────────┐
                         │  TextEncoder trait                      │
                         │  MockEncoder → BGE-M3 (future)          │
                         │  dimension = 8→512→768                  │
                         │                                         │
                         │  ┌──────────────────────────────┐       │
                         │  │         CangSea v2            │       │
                         │  │  SemanticStore (向量数据库)    │       │
                         │  │  + Legacy 64×64 matrix        │       │
                         │  └──────────────────────────────┘       │
                         │         ▲                  │             │
                         │   query │                  │ store       │
                         │         │                  ▼             │
                         │  ┌──────┴──────┐  ┌──────────────────┐  │
                         │  │  连山 (语义) │  │  周易 (语义)      │  │
                         │  │ V_goal      │  │  V_origin        │  │
                         │  │ V_obstacle  │  │  V_strategy      │  │
                         │  │ C_history   │  │         →        │  │
                         │  │     →       │  │  V_attitude      │  │
                         │  │ V_strategy  │  │  prompt_prefix   │  │
                         │  └──────┬──────┘  └────────┬─────────┘  │
                         │         │                  │             │
                         └─────────┼──────────────────┼─────────────┘
                                   │                  │
                                   ▼                  ▼
                         ┌─────────────────────────────────────┐
                         │          阳仪 (Yang)                 │
                         │  LlmBackend (生成 LLM)               │
                         │  LogitBias ← hybrid_deviation       │
                         │  Context  ← V_attitude (meta-prompt) │
                         └─────────────────────────────────────┘
```

### 2.1 归一化数据结构

所有三引擎与阴仪的交互都通过藏海 SemanticStore 的归一化接口：

```
每次"藏"阶段写入：
  SemanticEntry = {
    v_think:    Embedding,      // 思维编码向量
    v_goal:     Embedding,      // 目标向量
    v_obstacle: Embedding,      // 障碍向量
    v_strategy: Embedding,      // 策略向量
    v_attitude: Embedding,      // 姿态向量
    deviation:  f32,            // 混合偏离度
    reward:     f32,            // 经验回报
    timestamp:  u64,            // 时间戳
    immune:     bool,           // 是否免疫区
    merge_count: u32,           // 晶体化合并次数
  }
```

---

## 三、三引擎语义化升级

### 3.1 归藏：混合偏离度

```
hybrid_deviation(hamming, Some(cosine_dev), alpha)
  = alpha * hamming + (1-alpha) * cosine_dev

其中：
  hamming   = hamming_distance(state, origin) / 6        ∈ [0, 1]
  cosine_dev = 1 - cosine_similarity(v_current, v_origin)  ∈ [0, 2)
  alpha     = 0.5（默认，可在 Runtime 调整）
```

| 偏离度 | 含义 | 动作 |
|--------|------|------|
| < 0.3 | 紧密围绕问题焦点 | 无干预 |
| 0.3 ~ 0.7 | 有一定扩展但仍在范围内 | 温和 Logit-Bias |
| > 0.7 | 明显偏离 | 连山策略重规划 + 周易姿态调整 |
| > 0.9 | 完全脱离问题 | KV 回滚 + 强力 Logit-Bias + 免疫记忆 |

### 3.2 连山：语义策略导航

当前的连山是 6 步查表决策（固定规则）。路线C 中升级为语义模式：

```
连山语义化输入：
  v_goal     = encoder.encode(goal_text)
  v_obstacle = encoder.encode(obstacle_text)
  c_history  = 最近 N 次连山决策历史

连山语义化输出：
  v_strategy = blend(top-3 similar strategies from CangSea)
  projected  = nearest FangWei (7 方向之一)
  confidence = similarity score between input and best match

关键升级：
  · 策略不再固定——CangSea 中存储的历史成功策略被自动检索和加权融合
  · WaitGather 不再是死代码——免疫命中模糊时 (0.4 < cos < 0.6) 被激活
  · 反循环检测——3 次连续 Decompose → 自动升级为 Escalate
```

### 3.3 周易：语义姿态选择

当前的周易是 8 卦 Markov 链（固定生克权重）。路线C 中升级为语义模式：

```
周易语义化输入：
  v_origin   = encoder.encode(origin_text)
  v_strategy = 连山输出的策略向量

周易语义化输出：
  v_attitude  = interpolate between base attitude vectors weighted by v_strategy
  temperature = top-3 base attitudes temperature weighted average (0.3-1.2)
  pose        = projected Bagua (8 卦之一)
  prompt_prefix = pose-specific meta-prompt text
  attention_bias = optional attention redistribution hints

关键升级：
  · 姿态不再是固定 Markov 链——当前策略向量自动插值产生最适合的姿态
  · 温度不再是固定的——每卦一个语义空间，连续插值产生连续温度
  · prompt_prefix 自动选择——每卦对应一套元提示文本
```

---

## 四、异步流水线：太极双引擎

### 4.1 阴阳分工

| 维度 | 阳仪 (Yang) | 阴仪 (Yin) |
|:----|:-----------|:-----------|
| **角色** | 生成 LLM | 编码 Encoder |
| **大小** | 9B（理想） | 100-500M（轻量） |
| **实例数** | 1 | 1 |
| **任务** | 自回归 token 生成 | 持续编码 + 语义计算 |
| **与三引擎关系** | 受三引擎约束 | 为三引擎提供语义输入 |
| **负载** | 高（持续生成） | 低（仅编码语义段） |

### 4.2 异步并行

核心洞察：**阴仪编码上一个算子的文本时，阳仪已经在生成下一个算子。**

```
时间轴 →
阳仪：[生_start][生算子内容生成...][生_end][动_start][动算子内容生成...][动_end]...
阴仪：[等待]  [等待]              [编码生][计算偏离度][       等待        ][编码动]...
连山：[                                                        ][连山决策]...
周易：[                                              ][周易决策]...
```

感知延迟增加几乎为零：
- 唯一的串行瓶颈：算子结束 → 偏离度计算 → 反馈到下一个算子的 Logit-Bias（~10-20ms）
- 连山和周易的语义决策在后台异步完成，不影响主生成线程

### 4.3 三引擎协同流程（10步语义周天）

```
 1. 生 (Sprout):
    IF semantic_mode:
      → v_sprout = encoder.encode(goal + obstacle)
      → query CangSea → blend top-3 → project to Gua
    vm.state = vm.state.sheng()

 2. 动 (Diverge):
    vm.state = vm.state.dong()

 3. 归 (Return & Measure):
    → hybrid_deviation(hamming, cosine_dev, alpha)
    IF dev > 0.9: abort cycle, store immune

 4. 长 (Amplify):
    vm.state = vm.state.zhang()

 5. 连山介入:
    → v_goal + v_obstacle + C_history → v_strategy, FangWei

 6. 周易介入:
    → v_origin + v_strategy → v_attitude, temperature, prompt_prefix

 7. 育 (Nurture):
    vm.state = vm.state.yu()
    IF dev > 0.7: mid-stage prune

 8. 杀 (Prune):
    vm.state = vm.state.sha()
    → 物理 token 丢弃（不进入 history）

 9. 止 (Freeze):
    vm.state = vm.state.zhi()

10. 藏 (Store & Reset):
    → 同时写入 SemanticStore + Legacy Matrix
    → 投影 v_final → nearest Gua
    → IF reward < -0.5: push to immune_zone
    → vm.state = 0
```

**向后兼容**: `semantic_mode == false` 时回退到当前 3-phase 8-step 离散模式。

---

## 五、藏海 v2：语义向量数据库

### 5.1 双层存储架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         CangSea v2                                │
│                                                                   │
│  Legacy 64×64 Matrix (不变)    │  SemanticStore (新增)            │
│  weights: [[f32;64];64]       │                                  │
│  entries: Vec<CangSeaEntry>   │  ┌─ entries: Vec<SemanticEntry>  │
│  max_entries: 1000            │  │  常规经验库 (5000 条)          │
│                               │  │  → query_similar_*() 检索     │
│  → 快速离散查表                │  │  → merge_similar() 晶体化     │
│  → 向后兼容                   │  └────────────────────────────── │
│  → Gua 投影锚点               │                                  │
│                               │  ┌─ immune_zone: Vec<SemEntry>   │
│                               │  │  免疫记忆区 (1000 条)          │
│                               │  │  → 负 reward 永久隔离         │
│                               │  │  → query_immune_similar()     │
│                               │  │  → 不被常规驱逐               │
│                               │  └────────────────────────────── │
│                               │                                  │
│                               │  MergeStrategy: WeightedAverage   │
│                               │  容量 > 80% 或每 10 周天 → 合并  │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 免疫记忆区

负 reward 条目（偏离度 > 0.9 的失败模式）永久存入 immune_zone：
- 不被常规淘汰策略驱逐
- 连山检索时 query_immune_similar() → 填充 resist_forces → 影响方位决策
- 周易检索时 is_pattern_dangerous() → 偏好保守姿态（艮/坤, 低温度）

### 5.3 经验晶体化

高相似度条目自动合并为"经验晶体"——保留强信号，泛化弱信号。这是藏海的进化机制：
- 成功模式被晶体化 → 增强泛化能力
- 失败模式被晶体化 → 避免重复错误

---

## 六、演进路径

```
路线A（外挂约束）    路线B（模型内化）     路线C（全语义）
   归藏引擎           三引擎流水线          三引擎语义化
   归藏 ✅           归藏 ✅              归藏 ✅ → hybrid_deviation
   连山 🔧 代码就绪   连山 ✅ 接线完成      连山 ✅ → 语义向量
   周易 🔧 代码就绪   周易 ✅ 接线完成      周易 ✅ → 语义姿态
   阴仪 = 正则        阴仪 = 正则+原语      阴仪 = TextEncoder
   藏海 = 64×64      藏海 = 64×64         藏海 = 64×64 + SemanticStore
   0 训练              LoRA 微调            MockEncoder → 真实 ONNX

                      ←←← 自举进化飞轮 →→→→→→→→→→→→→→→→→→→
                      路线A/路线C 的质量过滤数据 → 路线B 微调
                      → 更好的三引擎 → 更高质量的数据 → 更好微调...
```

---

## 七、验证标准

### 7.1 单元测试验证

| 测试 | 验证内容 | 涉及模块 |
|------|---------|---------|
| `test_taiji_fallback` | semantic_mode=false → 行为与当前一致 | CangVM |
| `test_taiji_mock_hybrid` | MockEncoder(Focused) → hybrid < 0.2 | deviation |
| `test_taiji_mock_drifting` | 3 轮 → 偏差递增 | deviation |
| `test_taiji_lian_shan_wired` | 连山介入 → V_strategy 输出有效 | shanvm/semantic |
| `test_taiji_zhou_yi_wired` | 周易介入 → V_attitude + prompt_prefix 有效 | zhouvm/semantic |
| `test_taiji_experience_crystal` | CangSea 含全字段 + 晶体化 | cang_sea |
| `test_taiji_physical_discard` | 杀触发 → result.deviated=true, history 不含偏离文本 | xiang-llm |
| `test_taiji_immune_memory` | 负 reward → immune_zone, query 命中 | cang_sea |
| `test_taiji_merge_crystal` | 高相似度合并 → merge_count 递增 | cang_sea |

### 7.2 集成验证

| 里程碑 | 定义 |
|--------|------|
| M0：MockEncoder 验证 | 8 维 MockEncoder 驱动全 10 步语义周天 |
| M1：三引擎语义化 | 连山/周易 从规则查表升级为语义插值 |
| M2：异步流水线 | 阴阳双引擎并行运行 |
| M3：物理丢弃 | 杀触发时 token 不进入 history |
| M4：免疫记忆 | 失败模式存入 immune_zone 并影响决策 |
| M5：经验晶体化 | 高相似度经验自动合并 |
| M6：真实编码器 | 从 MockEncoder 切换到真实 ONNX 模型 |

---

## 八、关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Encoder 接口 | `TextEncoder` trait，Mock 先行 | 不绑定具体 ONNX 模型 |
| 向后兼容 | 所有新字段 `Option<T>` | zero-cost abstraction |
| 藏海写入 | 同时写入 Legacy + Semantic | 保持向后兼容，两条路径可独立验证 |
| 藏海读取 | Semantic 优先，Legacy fallback | 渐进迁移 |
| 物理丢弃 | 条件 `add_turn()` | 杀触发时 token 不进入 history |
| 免疫区 | 永不驱逐，仅在内部 merge | 防止危险模式被遗忘 |

---

> **详细实施计划见下文 §九。**
>
> **当前状态** → [00-总纲.md](./00-总纲.md)（三引擎实现现状表）

→ [路线A：阶段约束](./01-路线A-阶段约束.md)（已完成）  
→ [路线B：控制原语微调](./02-路线B-控制原语微调.md)（可选路线，硬件升级后执行）

---

## 九、工程实施计划

> 以下内容合并自原 `路线C-实施计划.md`。详细工程步骤、代码审计发现、风险缓解。

### 9.1 设计原则

1. **向后兼容**: 所有新字段为 `Option<T>`，无 encoder 时回退纯 Hamming + 查表
2. **Mock 先行**: MockEncoder 验证全架构 → 再集成真实 ONNX 模型
3. **功能对称，体量不对称**: 阳仪 4B 模型，阴仪 100-500M 轻量编码器
4. **保留现有代码**: ShanVM/ZhouVM 原 struct 不动，语义实现放在同 crate 的 `semantic.rs`
5. **Trait 在 core，实现分散**: 避免循环依赖，trait 对象注入 CangVM

### 9.2 实现步骤

#### Phase 1: Core Types — 太极基础 (`xiang-core`)

**创建: `crates/xiang-core/src/embedding.rs`**
- `Embedding` = `Vec<f32>` 类型别名
- `cosine_similarity(a, b) -> f32`
- `TextEncoder` trait: `encode(text) -> Embedding`, `dimension()`, `model_name()`
- `MockEncoder` struct: 确定性伪随机向量（默认 8 维）
- `MockEncoderMode`: `Focused`, `Drifting`, `Hallucinating`

**创建: `crates/xiang-core/src/semantic.rs`**
- `StrategyInput { v_goal, v_obstacle, c_history }` — 连山输入
- `StrategyOutput { v_strategy, projected: FangWei, confidence }` — 连山输出
- `SemanticDecision` trait: `decide(input) -> StrategyOutput`
- `AttitudeInput { v_origin, v_strategy }` — 周易输入
- `AttitudeOutput { v_attitude, temperature, pose: Bagua, prompt_prefix, attention_bias }`
- `AttitudeEncoder` trait: `encode_attitude(input) -> AttitudeOutput`

**修改: `crates/xiang-core/src/deviation.rs`**
- 新增 `hybrid_deviation(hamming, semantic_option, alpha) -> f32`
- 新增 `DeviationSource` 枚举

**修改: `crates/xiang-core/src/cang_sea.rs`**
- 新增 `SemanticEntry` struct 和 `SemanticStore` struct
- 核心方法: `store_semantic()`, `query_similar_*()`, `query_immune_similar()`, `merge_similar()`
- `CangSea` 新增 `semantic_store: Option<SemanticStore>` 字段

#### Phase 2: Mock Encoder (`xiang-encoder` crate)

**新建 crate: `crates/xiang-encoder/`** → `MockEncoder`: 实现 `TextEncoder`

#### Phase 3: 连山语义化 (`xiang-shanvm`)

**创建: `crates/xiang-shanvm/src/semantic.rs`**
- 7 个基础策略向量 `BASE_STRATEGIES`（对应 FangWei 7 方向）
- `SemanticLianShan<E: TextEncoder>` struct → mock k-NN + 加权插值策略

#### Phase 4: 周易语义化 (`xiang-zhouvm`)

**创建: `crates/xiang-zhouvm/src/semantic.rs`**
- 8 个基础姿态向量 `BASE_ATTITUDES`（对应 8 卦）
- `SemanticZhouYi<E: TextEncoder>` struct → V_origin + V_strategy 插值

#### Phase 5: CangVM 接线 (`xiang-cangvm`)

`CangVM` 新增字段: `semantic_origin`, `semantic_deviation`, `deviation_alpha`, `semantic_lianshan`, `semantic_zhouyi`, `text_encoder`
- `execute_lianshan_op()` — 替换原 no-op
- `execute_zhouyi_op()` — 替换原 no-op
- `current_deviation()` 改写 → 调用 `hybrid_deviation()`

#### Phase 6: 完整语义周天 — 10 步认知循环

`ZhouTianRunner::run_cycle()` 升级为 10 步语义循环（gated on `semantic_mode`）：
1. 生 → 编码 goal+obstacle → 检索藏海 → 设置启发态
2. 动 → 记录当前思维轨迹
3. 归 → hybrid_deviation（偏离 > 0.9 则终止）
4. 长 → 精炼目标向量
5. 连山介入 → 语义策略决策
6. 周易介入 → 语义姿态选择 + temperature + prompt_prefix
7. 育 → 偏离 > 0.7 触发中期剪除
8. 杀 → Gua 层级剪除
9. 止 → 边界凝固
10. 藏 → 写入 SemanticStore + Legacy Matrix

**向后兼容**: `semantic_mode == false` 时回退当前 3-phase 8-step。

#### Phase 7: LogitBias 升级 (`xiang-llm`)

`XiangLogitBias` 新增: `hamming_deviation`, `semantic_deviation`, `v_strategy`, `v_attitude`

#### Phase 8: 杀硬逻辑 — 物理 Token 丢弃 (`xiang-llm` + `xiang-experiments`)

杀触发时 token 物理丢弃，不进入上下文历史。重构 `generate_constrained_turn()` 为条件 add_turn 路径。

#### Phase 9: ExperimentRunner 集成 (`xiang-experiments`)

`ExperimentConfig` 新增: `use_encoder`, `encoder_dim`, `semantic_mode`, `merge_threshold`
CLI: `--encoder`, `--encoder-dim`, `--alpha`, `--semantic`

### 9.3 关键文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| CREATE | `crates/xiang-core/src/embedding.rs` | TextEncoder trait + MockEncoder |
| CREATE | `crates/xiang-core/src/semantic.rs` | SemanticDecision + AttitudeEncoder traits |
| MODIFY | `crates/xiang-core/src/deviation.rs` | hybrid_deviation() |
| MODIFY | `crates/xiang-core/src/cang_sea.rs` | SemanticStore + SemanticEntry |
| CREATE | `crates/xiang-encoder/` | 新 crate (MockEncoder) |
| CREATE | `crates/xiang-shanvm/src/semantic.rs` | SemanticLianShan |
| CREATE | `crates/xiang-zhouvm/src/semantic.rs` | SemanticZhouYi |
| MODIFY | `crates/xiang-cangvm/src/vm.rs` | 接线 + 10步语义循环 |
| MODIFY | `crates/xiang-llm/src/lib.rs` | LogitBias + 物理丢弃 |
| MODIFY | `crates/xiang-experiments/src/lib.rs` | 三引擎集成 |
| MODIFY | `Cargo.toml` | 添加 xiang-encoder 成员 |

### 9.4 代码审计发现

| 发现 | 影响 | 决策 |
|------|------|------|
| CangSea 全部字段 private | 验证了设计 | SemanticStore 作为独立公开扩展字段 |
| FangWei::WaitGather 是死代码 | 7方向中1个不可达 | 语义连山激活此路径（免疫模糊匹配）|
| LlmContext::add_turn() 无条件调用 | 与物理丢弃冲突 | Phase 8 条件 add_turn 重构 |
| ShanVM.decision_log 未被读取 | 历史决策闲置 | 语义连山反循环检测 |
| Gua::ORIGIN 注释已指向动态化 | 代码已为语义化准备 | Phase 5 set_semantic_origin() |

### 9.5 验证方法

**单元测试**: embedding.rs cosine 验证、deviation.rs hybrid 回退、cang_sea.rs 免疫区隔离与合并

**集成测试**:
- `test_taiji_fallback`: semantic_mode=false 行为一致
- `test_taiji_mock_hybrid`: MockEncoder(Focused) → hybrid < 0.2
- `test_taiji_mock_drifting`: 3轮偏差递增
- `test_taiji_lian_shan_wired`: 连山输出有效
- `test_taiji_zhou_yi_wired`: 周易输出有效
- `test_taiji_physical_discard`: 杀触发 token 不入 history
- `test_taiji_immune_memory`: 负 reward → immune_zone 命中
- `test_taiji_merge_crystal`: 高相似度合并

### 9.6 风险与缓解

| 风险 | 缓解 |
|------|------|
| `generate_constrained_turn` 签名变化影响调用者 | 签名不变，内部 self 状态管理 |
| SemanticStore 与 Legacy Matrix 数据不一致 | 每次藏同时写两边；检索 Semantic 优先 |
| MockEncoder 确定性不足 | 确定性哈希 → 单位向量 |
| CangVM 字段膨胀 | 全部 Option，semantic_mode=false 时零开销 |
| 免疫区无限增长 | immune_zone_max=1000，超限时内部 merge 不驱逐 |
