# 太极模型 (Taiji Model) — 三易全链路语义化实现计划

**版本：2.0** | **定位：本文档是 specs/路线C-实施计划.md 的副本，供 Qoder 上下文使用**

> 本文档与 [specs/路线C-实施计划.md](../specs/路线C-实施计划.md) 内容同步。  
> 三引擎架构的完整愿景见 [项目根 README.md](../README.md) 和 [specs/README.md](../specs/README.md)。  
> 详细工程步骤、代码审计发现、风险缓解见 [specs/路线C-实施计划.md](../specs/路线C-实施计划.md)。

## Context

### 当前状态

归藏系统的三层架构（归藏/连山/周易）中：
- **CangVM (归藏)**: 已集成到实验管线。偏离度用 6-bit 汉明距离（仅 7 个离散级别），origin 固定为 `Gua::ORIGIN`(0b111111)。启动可靠性已修复：`is_ready()` 改为实时健康检查 + 30 秒可配置词汇发现重试
- **ShanVM (连山)**: 已实现但未接线。`LianShanOp` 在 CangVM 中被 no-op（标注 "Phase 4"），6 步查表决策输出 7 个固定方向
- **ZhouVM (周易)**: 已实现但未接线。`ZhouYiOp` 在 CangVM 中被 no-op（标注 "Phase 4"），8 状态 Markov 网格，每卦固定温度
- **CangSea**: 64×64 Hebbian 权重矩阵，存储 6-bit Gua 状态转移，最大 1000 条，FIFO 淘汰

### 目标：太极模型全链路语义化

通过引入**阴仪（TextEncoder）**使整个三易系统从"离散查表"跃升为"语义驱动"：

- **阳仪 (Yang)**: 生成 LLM — 负责 token 生成
- **阴仪 (Yin)**: 压缩 Encoder — 持续监控，计算语义偏离，驱动连山策略与周易姿态
- **归藏**: hybrid_deviation = α·hamming + (1-α)·cosine
- **连山**: V_goal + V_obstacle + C_history → V_strategy（语义策略分类器）
- **周易**: V_origin + V_strategy → V_attitude（认知姿态编码器）
- **藏海**: 从 64×64 矩阵升级为语义向量数据库

## 架构概览

```
                         ┌── 阴仪 (Yin) ──────────────────────────┐
                         │  TextEncoder trait                      │
                         │  MockEncoder → BGE-M3 (future)          │
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

## 设计原则

1. **向后兼容**: 所有新字段为 `Option<T>`，无 encoder 时回退纯 Hamming + 查表
2. **Mock 先行**: MockEncoder 验证全架构 → 再集成真实 ONNX 模型
3. **功能对称，体量不对称**: 阳仪 9B 大模型，阴仪 100-500M 轻量编码器
4. **保留现有代码**: ShanVM/ZhouVM 原 struct 不动，语义实现放在同 crate 的 `semantic.rs`
5. **Trait 在 core，实现分散**: 避免循环依赖，trait 对象注入 CangVM

## 实现步骤

### Phase 1: Core Types — 太极基础 (`xiang-core`)

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
- 新增 `SemanticEntry` struct: `{v_think, v_goal, v_obstacle, v_strategy, v_attitude, deviation, reward, timestamp, immune_count, merge_count, crystal_generation}`
- 新增 `SemanticStore` struct:
  - `entries: Vec<SemanticEntry>` — 常规经验库
  - `immune_zone: Vec<SemanticEntry>` — **免疫记忆区**（负 reward 经验永久隔离，不被常规驱逐）
  - `max_entries: usize`, `immune_zone_max: usize`
  - `merge_strategy: MergeStrategy` — `WeightedAverage | MostRewarded | Centroid`
- 核心方法:
  - `store_semantic()`: reward < -0.5 → 同时推入 immune_zone；容量 > 80% → 触发合并
  - `query_similar_*()`: cosine 相似度检索（策略/姿态/通用）
  - `query_immune_similar(embedding, threshold)` — 检索免疫区，检测已知危险模式
  - `is_pattern_dangerous(embedding, threshold) -> bool` — 快速判断当前思维是否接近免疫模式
  - `merge_similar(threshold)` — **冗余合并**：高相似度条目加权融合为"经验晶体"，保留强信号，泛化弱信号
  - `merge_immune_zone()` — 免疫区内部合并（免疫条目之间可合并，但不与常规区混合）
- `CangSea` 新增 `semantic_store: Option<SemanticStore>` 字段

**修改: `crates/xiang-core/src/lib.rs`** — 导出 `pub mod embedding; pub mod semantic;`

### Phase 2: Mock Encoder (`xiang-encoder` crate)

**新建 crate: `crates/xiang-encoder/`**
- 注册到 workspace `Cargo.toml`

**创建: `crates/xiang-encoder/src/mock.rs`**
- `MockEncoder`: 实现 `TextEncoder`，文本哈希 → 确定性单位向量
- `MockEncoderMode`: Focused（低偏差）/ Drifting（递增偏差）/ Hallucinating（高偏差）

### Phase 3: 连山语义化 (`xiang-shanvm`)

**创建: `crates/xiang-shanvm/src/semantic.rs`**
- 7 个基础策略向量 `BASE_STRATEGIES`（对应 FangWei 的 7 个方向）
- `SemanticLianShan<E: TextEncoder>` struct
- 实现 `SemanticDecision` trait: mock k-NN + 加权插值策略
- 决策逻辑: 从 CangSea 检索类似障碍-策略对 → 历史 reward 加权 → 输出 V_strategy

### Phase 4: 周易语义化 (`xiang-zhouvm`)

**创建: `crates/xiang-zhouvm/src/semantic.rs`**
- 8 个基础姿态向量 `BASE_ATTITUDES`（对应 8 卦）
- `SemanticZhouYi<E: TextEncoder>` struct
- 实现 `AttitudeEncoder` trait: V_origin + V_strategy → 插值 → V_attitude
- 8 套 prompt_template（每卦一个元提示文本）
- temperature: top-3 基向量温度加权平均

### Phase 5: CangVM 接线 — 连山/周易接入 (`xiang-cangvm`)

**修改: `crates/xiang-cangvm/src/vm.rs`**

`CangVM` 新增字段:
- `semantic_origin: Option<SemanticOrigin>`
- `semantic_deviation: Option<f32>`
- `deviation_alpha: f32` (默认 0.5)
- `semantic_lianshan: Option<Box<dyn SemanticDecision>>`
- `semantic_zhouyi: Option<Box<dyn AttitudeEncoder>>`
- `text_encoder: Option<Box<dyn TextEncoder>>`

新方法:
- `set_semantic_origin(text, embedding)` — 动态设定语义原点
- `run_semantic_lianshan(v_goal, v_obstacle) -> Option<StrategyOutput>`
- `run_semantic_zhouyi(v_origin, v_strategy) -> Option<AttitudeOutput>`
- `execute_lianshan_op()` — 替换原 no-op
- `execute_zhouyi_op()` — 替换原 no-op
- `current_deviation()` 改写 → 调用 `hybrid_deviation()`

### Phase 6: 完整语义周天 — 10 步认知循环 (`xiang-cangvm`)

**修改: `crates/xiang-cangvm/src/vm.rs` — `ZhouTianRunner`**

新增字段:
- `pub question_text: String`, `pub goal_text: String`, `pub obstacle_text: Option<String>`
- `pub semantic_mode: bool` (默认 false → 回退 3-phase 8-step)
- `pub last_v_strategy: Option<Embedding>`, `pub last_v_attitude: Option<Embedding>`
- `pub last_v_origin: Option<Embedding>`
- `pub text_encoder: Option<Box<dyn TextEncoder>>`

`run_cycle()` 升级为 10 步语义循环（gated on `self.semantic_mode`）:

```
 1. 生 (Sprout):
    IF semantic_mode:
      → v_sprout = encoder.encode(format!("goal: {} | obstacle: {:?}", goal_text, obstacle_text))
      → query CangSea.semantic_store for top-3 similar experiences
      → 若命中: blend v_sprout with weighted historical vectors
      → 投影 v_sprout → nearest Gua, set vm.state 为启发态
    vm.state = vm.state.sheng()

 2. 动 (Diverge):
    vm.state = vm.state.dong()
    IF semantic_mode:
      → 记录当前 v_thought = encoder.encode("current thought trajectory")  // 藏到日志

 3. 归 (Return & Measure):
    IF semantic_mode:
      → v_current = encoder.encode("state summary from output_log")
      → sem_dev = 1.0 - cosine_similarity(v_current, last_v_origin)
      → dev = hybrid_deviation(hamming, Some(sem_dev), alpha)
    ELSE:
      → dev = vm.current_deviation()
    vm.deviation_log.push((vm.cycle_count, dev))
    
    IF dev > refocus_threshold (0.9):
      → vm.state = vm.state.sha(); vm.state = vm.state.cang()
      → IF semantic_mode: store_semantic(reward=-0.5), push immune_zone
      → return (abort cycle)

 4. 长 (Amplify):
    vm.state = vm.state.zhang()
    IF semantic_mode:
      → refine goal embedding: v_goal = encoder.encode(goal_text + " elaborated")
      → store v_goal as last_known_v_goal

 5. 连山介入 (Lianshan Intervention):
    IF semantic_mode AND semantic_lianshan.is_some():
      → v_goal = encoder.encode(goal_text)
      → v_obstacle = encoder.encode(obstacle_text or "")
      → strategy_out = semantic_lianshan.decide(StrategyInput { v_goal, v_obstacle, c_history: todo!() })
      → self.last_v_strategy = Some(strategy_out.v_strategy)
    // ELSE: no-op (保持现有 behavior)

 6. 周易介入 (ZhouYi Intervention):
    IF semantic_mode AND semantic_zhouyi.is_some():
      → v_origin = last_v_origin
      → v_strategy = last_v_strategy
      → attitude_out = semantic_zhouyi.encode_attitude(AttitudeInput { v_origin, v_strategy })
      → self.last_v_attitude = Some(attitude_out.v_attitude)
      → prompt_prefix = attitude_out.prompt_prefix  // 传入 LlmContext
    // ELSE: no-op

 7. 育 (Nurture):
    vm.state = vm.state.yu()
    IF dev > deviation_threshold (0.7):
      → vm.state = vm.state.sha()  // mid-stage prune

 8. 杀 (Prune):
    vm.state = vm.state.sha()
    // NOTE: 物理丢弃在 Phase 8 中实现，此处为 Gua 层级的 sha 算子

 9. 止 (Freeze):
    vm.state = vm.state.zhi()

10. 藏 (Store & Reset):
    · IF semantic_mode:
        → v_final = encoder.encode("final thought state")
        → dev = hybrid_deviation(hamming(state,origin), Some(cosine_dev), alpha)
        → SemanticEntry: { v_think: v_final, v_goal, v_obstacle, v_strategy, v_attitude, deviation, reward, ... }
        → store_semantic(entry) in CangSea
        → 同时投影 v_final → nearest Gua，存入 legacy matrix（保持向后兼容）
    · vm.state = vm.state.cang()  // → 0
    · IF dev < aligned_threshold: reward +0.5, cang_aligned++
    · ELSE: reward -0.3, cang_deviated++
```

**向后兼容**: `semantic_mode == false` 时，所有 encoder 调用被跳过，行为与当前 3-phase 8-step 完全一致。

### Phase 7: LogitBias 升级 (`xiang-llm`)

**修改: `crates/xiang-llm/src/lib.rs`**
- `XiangLogitBias` 新增: `hamming_deviation`, `semantic_deviation`, `v_strategy`, `v_attitude`
- `GenerationParams` 新增: `semantic_deviation`, `prompt_prefix`
- `LlmContext::generate_constrained_turn()` 新增参数
- `BiasLogEntry` 新增语义字段

### Phase 8: 杀硬逻辑 — 物理 Token 丢弃 (`xiang-llm` + `xiang-experiments`)

白皮书要求：杀触发时，token **物理丢弃**，不进入上下文历史；后续生成被强制引导回 V_origin。

**现状**: `LlmContext::generate_constrained_turn()` (lib.rs:364-383) 在调用 `backend.generate()` 后**无条件**执行 `self.add_turn(user_input, &r.text)`。必须重构为条件路径。

**修改: `crates/xiang-llm/src/lib.rs` — `GenerationResult`**
- 新增 `pub deviated: bool` — 标记该轮生成是否因偏离被截断

**修改: `crates/xiang-llm/src/lib.rs` — `LlmContext`**
- 新增 `pub origin_guidance: Option<String>` — 偏离后的强制焦点引导语
- 新增 `pub skipped_turns: Vec<(String, String)>` — 保留被丢弃的轮次供调试/免疫
- 重构 `generate_constrained_turn()` 流程（当前代码无条件 add_turn，需拆分为条件路径）:
  ```
  1. 检查 self.origin_guidance: 若 Some，将其注入 user_input 前
     → 实际 user_prompt = "[元指令] {guidance}\n\n{original_input}"
  2. 构建 XiangLogitBias（不变）
  3. backend.generate(&params) → result
  4. 检查 result.stop_reason == StopReason::Deviated:
     若 true:
       → 设置 self.origin_guidance = Some("请将焦点回归核心问题。上一次思考方向偏离既定目标。")
       → self.skipped_turns.push((user_input.clone(), result.text.clone()))
       → **不调用 add_turn()** — token 物理丢弃
       → 设置 result.deviated = true
       → increment focus_interventions
     若 false:
       → 清除 self.origin_guidance = None
       → 正常调用 self.add_turn(user_input, &result.text)
  ```
- `generate_unconstrained_turn()`: 不变（无偏差干预逻辑）

**修改: `crates/xiang-llm/src/http_backend.rs`** (line 137-142)
- `bias_to_logit_bias()` 返回 `force_stop=true` 时，当前已返回 `StopReason::Deviated`（无需改 stop_reason）
- 需确保返回的 `GenerationResult.deviated = true`（新增字段）

**修改: `crates/xiang-llm/src/llama_backend.rs`**
- 同样检查 force_stop 路径，设置 `result.deviated = true`

**修改: `crates/xiang-experiments/src/lib.rs` — `feedback_to_vm()`**
- 新增参数 `deviated: bool`
- deviated == true 或 D > 0.7 → `store_semantic(reward = -0.8)` 同时推入 `immune_zone`
- 正常轮次 → `store_semantic(reward = +0.3)` 仅存入常规区

**重要**: `generate_constrained_turn()` 的调用者（ExperimentRunner）需传递 origin_guidance 状态。当前 signature 为:
```rust
pub fn generate_constrained_turn<B: LlmBackend>(
    &mut self, backend: &mut B, user_input: &str, max_tokens: u32,
    temperature: TemperatureMode, vm_state: Gua, deviation: f32,
    sha_count: u32, off_focus_ids: Vec<u32>, divergent_ids: Vec<u32>, eos_id: u32,
) -> Result<GenerationResult, LlmError>
```
签名字段不变，origin_guidance 通过 `self` 内部状态管理。

### Phase 9: ExperimentRunner 集成 (`xiang-experiments`)

**修改: `crates/xiang-experiments/src/lib.rs`**
- `ExperimentConfig` 新增: `use_encoder`, `encoder_dim`, `semantic_mode`, `merge_threshold`, `immune_zone_max`
- 新增 `create_encoder()` → `MockEncoder`
- 新增 `create_semantic_lianshan()` / `create_semantic_zhouyi()`
- `run_experiment()`: 
  - 编码问题 → set_semantic_origin，注入 semantic engines 到 CangVM
  - 每轮调用语义连山/周易
  - 生成后检查 `result.deviated` → 触发物理丢弃 + 免疫存储
  - 每 10 个完整周天 → 调用 `cang_sea.semantic_store.merge_similar()`
- `ExperimentMetrics` 新增: `semantic_deviations`, `strategies_used`, `attitudes_used`, `immune_hits`, `merge_events`

**修改: `crates/xiang-experiments/src/main.rs`**
- CLI: `--encoder`, `--encoder-dim`, `--alpha`, `--semantic`

**修改: workspace 根 `Cargo.toml`** — 添加 `xiang-encoder` 成员

## 藏海升级：从 64×64 矩阵到语义向量数据库

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

写入: ZhouTianRunner 藏阶段
     → reward >= 0: store_semantic() → entries
     → reward < -0.5 (杀触发): store_semantic() → entries AND immune_zone
     同时投影 v_think → 最近 Gua，存入 legacy matrix
     
读取: 连山 decide() → query_similar_strategies(v_goal, v_obstacle, top_k=5)
                    → query_immune_similar(v_obstacle, 0.75) → resist_forces
      周易 encode_attitude() → query_similar_attitudes(v_origin, top_k=3)
                             → is_pattern_dangerous() → 保守姿态

免疫交互: 连山用免疫命中填充 ZhiForces.resist_forces
          → 自然引流为 NavigateAround / Abort 决策
          周易用 is_pattern_dangerous → 偏好保守姿态 (艮/坤, 低温度)
```

## 关键文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| CREATE | `crates/xiang-core/src/embedding.rs` | TextEncoder trait + MockEncoder + cosine math |
| CREATE | `crates/xiang-core/src/semantic.rs` | SemanticDecision + AttitudeEncoder traits + I/O types |
| MODIFY | `crates/xiang-core/src/deviation.rs` | hybrid_deviation() |
| MODIFY | `crates/xiang-core/src/cang_sea.rs` | SemanticStore + SemanticEntry |
| MODIFY | `crates/xiang-core/src/lib.rs` | 导出新模块 |
| CREATE | `crates/xiang-encoder/` | 新 crate (MockEncoder) |
| CREATE | `crates/xiang-shanvm/src/semantic.rs` | SemanticLianShan 实现 |
| CREATE | `crates/xiang-zhouvm/src/semantic.rs` | SemanticZhouYi 实现 |
| MODIFY | `crates/xiang-cangvm/src/vm.rs` | CangVM 接线 + ZhouTianRunner 10步语义循环 |
| MODIFY | `crates/xiang-llm/src/lib.rs` | LogitBias + GenerationResult.deviated + LlmContext 物理丢弃 |
| MODIFY | `crates/xiang-llm/src/http_backend.rs` | force_stop → deviated = true |
| MODIFY | `crates/xiang-experiments/src/lib.rs` | ExperimentRunner 集成三易语义引擎 + 免疫 + 合并 |
| MODIFY | `crates/xiang-experiments/src/main.rs` | CLI 标志 |
| MODIFY | `Cargo.toml` | 添加 xiang-encoder 成员 |

## 验证方法

1. **单元测试**:
   - `embedding.rs`: cosine 对相同/正交/相反/零向量
   - `deviation.rs`: hybrid 各种 alpha 和 None 回退
   - `cang_sea.rs`: SemanticStore 存储/检索/淘汰；免疫区隔离；merge_similar 加权融合
   - `xiang-shanvm/semantic.rs`: Mock 决策输出确定性；免疫命中→resist_forces
   - `xiang-zhouvm/semantic.rs`: Mock 姿态编码确定性；is_pattern_dangerous→保守姿态

2. **集成测试**:
   - `test_taiji_fallback`: use_encoder=false → 行为与旧版一致
   - `test_taiji_mock_hybrid`: MockEncoder(Focused) → hybrid < 0.2
   - `test_taiji_mock_drifting`: 3 轮→偏差递增
   - `test_taiji_lian_shan_wired`: 连山介入 → V_strategy 输出有效
   - `test_taiji_zhou_yi_wired`: 周易介入 → V_attitude + prompt_prefix 有效
   - `test_taiji_experience_crystal`: CangSea SemanticEntry 含全字段
   - **`test_taiji_physical_discard`**: 杀触发 → result.deviated=true, history 不含偏离文本, origin_guidance 被注入
   - **`test_taiji_immune_memory`**: 负 reward 条目推入 immune_zone, query_immune_similar 命中已知危险模式
   - **`test_taiji_merge_crystal`**: 高相似度条目合并为经验晶体, merge_count 递增, reward 取最大值

3. **编译验证**: `cargo build --workspace`
4. **全测试**: `cargo test --workspace`

## 代码审计发现与设计决策

经过对现有代码库的全面审查，以下是影响实现计划的关键发现及设计决策：

### 发现 1: CangSea 全部字段 private

`CangSea` 的 `weights`、`entries`、`max_entries`、`clock` 均为 private 字段。这验证了计划的设计：新增的 `pub semantic_store: Option<SemanticStore>` 是 CangSea 上唯一的公开扩展字段。SemanticStore 独立管理自己的容量和淘汰策略，不与 legacy Hebbian 矩阵耦合。两条路径并存：
- Legacy: `store(from: Gua, to: Gua, reward)` → `weights[from][to]` 更新
- Semantic: `store_semantic(entry: SemanticEntry)` → `semantic_store.entries` 追加

### 发现 2: FangWei::WaitGather 是死代码

`FangWei` 枚举定义了 7 个方向，但 `ShanVM::make_decision()` 中的决策表**从不返回** `WaitGather`。该变体在 12 个决定规则中被跳过。语义连山引擎可以激活此路径：当 `query_immune_similar()` 返回模糊匹配（0.4 < cosine < 0.6）时，系统选择 `WaitGather`——"这个模式有点熟悉但不完全确定，先收集更多信息"。这完美复用了现有 enum 而不断增变体。

### 发现 3: LlmContext::add_turn() 无条件调用

当前 `generate_constrained_turn()` 在 line 383 无条件执行 `self.add_turn(user_input, &r.text)`，即使 `stop_reason == Deviated`。这直接与"杀硬逻辑"要求冲突——物理丢弃要求 token 不进入 history。Phase 8 的"条件 add_turn"重构正是为了解决这个确切的代码路径缺陷。

### 发现 4: ShanVM.decision_log 未被决策逻辑读取

`decision_log: Vec<LianShanDecision>` 已积累完整的历史决策，但 `make_decision()` 从不读取它。这为"反循环"提供了天然基础：语义连山可在观察倒数 N=3 个决策均为 `Decompose` 时自动升级为 `Escalate`，防止无限分解循环。

### 发现 5: ZhouGrid 权重无界增长

`apply_shengke()` 的权重无上限/下限，理论上可无限增长（虽软最大归一化限制其影响，但绝对值膨胀）。纯离散 ZhouGrid 不受此影响（权重仅用于 softmax 采样），但未来若引入基于权重幅度的语义偏置，需添加 `MAX_WEIGHT = 10.0` / `MIN_WEIGHT = -5.0` 边界。此项不阻塞当前计划。

### 发现 6: Gua::ORIGIN 注释已指向动态化

`Gua::ORIGIN` 的 doc comment 已明确写道："此值应由当前问题的语义焦点动态设置，而非固定全阳"。这与 Phase 5 的 `set_semantic_origin()` 设计完全一致——代码基础已经为此做好了注解准备。

### 设计决策: Semantic 与 Legacy 的交互协议

```
写入操作:
  每次藏阶段: 同时写入 BOTH SemanticStore AND Legacy Matrix
    → semantic: full embedding + metadata
    → legacy:   投影 v_think → 最近 Gua, store(from, to, reward)
  
读取操作:
  连山: prefer Semantic query_similar_strategies(), fallback 到 legacy hebbian_sample()
  周易: prefer Semantic query_similar_attitudes(), fallback 到 legacy ZhouGrid.next_pose()
  生:  prefer Semantic query_similar_experiences(), fallback 到 legacy hebbian_sample()
  
内存权衡:
  SemanticStore capacity: 5000 entries × 512-dim × f32 = ~10MB
  Legacy matrix: 64×64 × f32 = ~16KB
  Total: ~10MB (可接受，尤其是 Mock 仅 8-dim = ~160KB)
```

### 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| `LlmContext.generate_constrained_turn` 签名变化影响调用者 | 低 | 签名不变，内部 self 状态管理 origin_guidance |
| SemanticStore 与 Legacy Matrix 数据不一致 | 中 | 每次藏同时写两边；检索时 Semantic 优先，Legacy fallback |
| MockEncoder 确定性不足导致测试不稳定 | 低 | 确定性哈希 → 单位向量，相同输入永远相同输出 |
| CangVM 字段膨胀（+5 new Option fields） | 低 | 全部 Option，零开销在 semantic_mode=false 时 |
| 免疫区无限增长 | 中 | immune_zone_max=1000，超限时对内部做 merge 而非驱逐（永不丢失危险记忆） |
