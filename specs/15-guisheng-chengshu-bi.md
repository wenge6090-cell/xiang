# 归藏生成树闭环 — 观测层、干预层、管理层设计

> 版本：1.0 | 生成日期：2026-05-27
>
> 本文档定义归藏引擎从"纯位运算状态机"升级为"完整的认知约束闭环"所需的所有设计决策。
> 以第一性原理（[10-第一性原理审视.md](./10-第一性原理审视.md)）为锚点。

---

## 〇、核心命题

**三易不是消灭幻觉——三易是在给涌现提供轨道。**

LLM 的幻觉与涌现同源，都来自连续向量空间的统计生成。杀死幻觉的方式必然同时杀死涌现。
三易不压制模型的内容选择——它在周围搭一个可变的"栅栏"：

| 策略 | 做法 | 对涌现的影响 |
|:----|:----|:----------|
| 压制式（主流） | Token 概率打到 -∞ | 模型变弱，不敢创造 |
| 引导式（三易） | 改变方向重心 ±1.0 | 模型在方向内自由涌现 |

---

## 一、三层执行架构

### 1.1 整体视图

```
用户输入（锚点：问题焦点）
    │
    ▼
┌─────────────────────────────────────────────────┐
│           周易 (ZhouVM) — 注意力姿态层             │
│                                                    │
│  功能：设定注意力的"工作模式"                       │
│  对应 LLM 内建机制：标准 Self-Attention（全连接）    │
│                                                    │
│  乾(创造)→温度1.2→注意力更敢于远距离跳转             │
│  离(分析)→温度0.5→注意力更精确收敛                  │
│  艮(审慎)→温度0.3→注意力最集中，不易漂移             │
│                                                    │
│  它不计算注意力——它改变注意力的"行为风格"             │
└──────────────────────┬──────────────────────────┘
                       │ 温度 + 姿态
                       ▼
┌─────────────────────────────────────────────────┐
│       归藏 (CangVM) — 注意力阶段干预层             │
│                                                    │
│  功能：标记当前认知阶段，实时干预 token 概率分布     │
│  对应 LLM 外建机制：带算子方向引导的 Logit-Bias     │
│                                                    │
│  生（萌发）→偏置重心偏向探索性 token 池             │
│  动（发散）→偏置重心偏向多角度 token 池             │
│  长（聚焦）→偏置重心偏向收敛性 token 池             │
│  育（构建）→偏置重心偏向结构性 token 池             │
│                                                    │
│  干预强度：±1.0 以内（方向引导，非内容压制）         │
│  它不压制"错误 token"——它改变 token 空间的采样重心   │
└──────────────────────┬──────────────────────────┘
                       │ 偏离度信号
                       ▼
┌─────────────────────────────────────────────────┐
│       连山 (ShanVM) — 注意力策略导航层             │
│                                                    │
│  功能：遇到障碍时切换注意力策略路径                 │
│  对应 LLM 类比：稀疏注意力（选择性关注方向）         │
│                                                    │
│  硬推(PushThrough)     → 保持当前注意力方向不变     │
│  绕行(NavigateAround)  → 转移注意力到替代路径      │
│  分解(Decompose)       → 切分注意力为子任务        │
│  升级(Escalate)        → 扩展注意力范围            │
│  放弃(Abort)           → 终止当前注意力路径        │
│  等待(WaitGather)      → 收缩注意力，等待信息      │
│                                                    │
│  它不计算注意力——它决定注意力该往哪个策略方向走      │
└─────────────────────────────────────────────────┘
```

### 1.2 三层与 LLM 注意力的映射

| LLM 内部（连续，不可读） | 三易（离散，可读可验证） | 引擎 |
|:----------------------|:---------------------|:----|
| Self-Attention 全连接注意力 | 注意力姿态层——设定"工作模式" | 周易 ZhouVM |
| 注意力阶段隐式切换 | 注意力阶段层——打阶段标签 + token 级干预 | 归藏 CangVM |
| 注意力策略隐式调整 | 注意力策略层——遇到障碍切换方向 | 连山 ShanVM |

**三层不替代注意力——三层是注意力的白箱化接口。**

### 1.3 三层协同防目标漂移

| 漂移深度 | 周易 | 归藏 | 连山 | 管理层 |
|:--------|:----|:----|:----|:------|
| token 级 | 降低温度 | 压制偏离方向 token（±0.5） | — | — |
| 句子级 | 收敛姿态 | 归观测到偏离持续上升 | 激活策略，绕行/分解 | 杀：裁该轮 |
| 段落级 | 艮（审慎） | 持续 high deviation | 升级/放弃 | 藏：清空重启 |

---

## 二、观测层：归 + HanziMap

### 2.1 设计原则

> 观测任何时候都接上。干预只做方向引导，不做内容压制。
> 执行只处理已发生的偏离，不预防未发生的。

**归是八气算子中唯一不做功的算子**。它不修改 S，只测量偏离度。
它是意识的自反性——"看自己流到哪里了"。

### 2.2 两种偏离度

| 偏离度 | 计算方式 | 范围 | 状态 |
|:------|:--------|:----|:----|
| 形式偏离度 | Hamming(state, origin) / 6.0 | [0, 1] | ✅ 已实现 |
| 语义偏离度 | hybrid: alpha·Hamming + (1-alpha)·cosine(Hanzi 投影指纹) | [0, 1] | 🔧 待接线 |

### 2.3 EmbeddingObserver 管道

```
LLM 解码每一步
    ↓
输出嵌入 [n_embd 维]
    ↓
HanziMap.map_top_k(emb, K) → 汉字序列 ["探","索","发","现"]
    ↓
EmbeddingObserver.observe(emb)
    ├── → 语义指纹序列 [汉字...]
    ├── → 与锚点的余弦偏离度
    └── → 五行元素分布
           ↓
CangVM.set_semantic_deviation() → hybrid_deviation() → judge()
```

### 2.4 观测 vs 干预 vs 管理的分离

| | 归 + HanziMap（观测） | 生/动/长/育（干预） | 杀/止/藏（管理） |
|:--|:-------------|:------------------------|:--------|
| 时间 | 每步生成后 | 每步生成前 | 整轮生成后 |
| 操作 | 只记录，不修改 | 改 token 概率分布 | 改对话历史 |
| 对模型影响 | 无 | ±1.0 方向引导 | 裁剪已偏离内容 |
| 前提 | 无——始终接上 | 需要观测信号 | 需要观测信号 |

**观测层必须最先接上。** 因为如果不观测到偏离，干预和执行就没有启动信号。

---

## 三、干预层：算子差异化 Logit-Bias

### 3.1 当前问题

当前 `XiangLogitBias` 对所有算子使用同一套 token 压制策略（off_focus / divergent），
丢失了算子阶段的差异化引导能力。v3.5 虽然调低了偏置强度，但仍然是"一锅烩"。

### 3.2 算子专属方向引导池

每个算子定义自己的 Positive 和 Negative token 池：

| 算子 | 阶段语义 | Positive token 倾向 | Negative token 倾向 |
|:----|:--------|:------------------|:------------------|
| **生** | 探索·提问·起始 | 也许、可能、方向、假设、探索、尝试、如何、是否 | 因此、总之、证明、结论、第一步 |
| **动** | 扩展·连接·发散 | 但是、然而、另一方面、相比之下、此外、同时 | 也就是说、即、简而言之、所以 |
| **长** | 深入·聚焦·收敛 | 因此、这表明、因为…所以、由此可见、进一步 | 另外、此外、换个角度、同时 |
| **育** | 构建·结构化·规划 | 第一步、其次、最后、综上所述、第一、第二 | 某种、也许、大概、可能、类似 |

### 3.3 偏置强度规则

```
基础强度:          ±0.5（所有算子默认）
偏离度调节:        ±0.3 × deviation（偏离时适当加强方向引导）
连山策略调节:      ±0.2 × strategy_factor（策略切换时微调）
最大强度上限:      ±1.0（硬上限——防止过度压制杀死涌现）
```

### 3.4 与连山策略的协同

- `PushThrough` → 所有 Positive bias +0.2（鼓励继续当前方向）
- `NavigateAround` → 所有 Negative bias +0.2（加强偏离抑制）
- 连山只调节 bias 强度，不改变 token 池构成

---

## 四、管理层：杀/止/藏 上下文操作

### 4.1 设计原则

> 杀/止/藏是压缩树，只做上下文管理，不做语义判断。

### 4.2 三级上下文管理

#### 第一级：对话历史管理（即刻可做）

| 算子 | 触发条件 | 操作 |
|:----|:--------|:----|
| **杀** | 偏离度 > 0.7 且持续多轮未恢复 | 删除最近 1-2 轮对话历史 |
| **止** | 偏离度 > 0.5 但未触发杀 | 注入"冻结标记"，系统提示词追加"请基于已有信息做结论" |
| **藏** | 周天循环完成 | 提取本周期摘要保留，清空消息列表 |

CangVM 新增信号输出枚举：

```rust
pub enum MetabolismSignal {
    Sha(u32),          // 杀 N 轮
    Zhi,               // 冻结上下文
    Cang { summary: String },  // 清空并保留摘要
}
```

#### 第二级：KV Cache 管理（需要 llama.cpp，P2）

```
每轮生成使用独占的 seq_id
  → kv_cache_snapshot() 保存当前 seq_id
  → 如果 judge() 返回 Rollback → 删除当前 seq_id 对应 KV cache
  → 下一轮从快照状态开始

HttpBackend 回退方案：提示词重打包
  → 从 history 移除偏离轮次 → 重新构建完整 prompt
  → 代价更高但功能等价
```

#### 第三级：Gua → 上下文操作的映射

```rust
// 在 CangVM::execute_phase_operator() 中新增信号输出
"杀" => {
    self.state = self.state.sha();
    self.sha_count += 1;
    self.metabolism_signal = Some(MetabolismSignal::Sha(2));
}
"止" => {
    self.state = self.state.zhi();
    self.metabolism_signal = Some(MetabolismSignal::Zhi);
}
"藏" => {
    // ... 现有藏海存储逻辑 ...
    let summary = self.metabolism.section();
    self.metabolism_signal = Some(MetabolismSignal::Cang { summary });
    self.state = self.state.cang();
}
```

### 4.3 信号处理流程

```
CangVM::execute_phase_operator()
    ↓
metabolism_signal 被设置
    ↓
回到 ConstrainedEngine::generate()
match signal {
    Sha(n)     → self.crop_history(n)
    Zhi        → self.freeze_context()
    Cang(s)    → self.reset_context(s)
}
```

---

## 五、闭环数据流总图

```
┌──────────────────────────────────────────────────────────────────┐
│                     ConstrainedEngine::generate()                   │
│                                                                      │
│  1. ZhouVM: select_zhou_pose() → temperature                        │
│                                                                      │
│  2. CangVM: current_operator() → "生"/"动"/"长"/"育"                │
│                                                                      │
│  3. CangVM: execute_phase_operator(&operator) → Gua 切换             │
│     └── 可能设置 metabolism_signal                                  │
│                                                                      │
│  4. CangVM: current_deviation() → hybrid_deviation                   │
│     ├── Hamming deviation（始终可用）                                 │
│     └── Semantic deviation（若 EmbeddingObserver 已注入）             │
│                                                                      │
│  5. deviation > 0.5 ? → ShanVM obstacle navigation                  │
│                                                                      │
│  6. XiangLogitBias: 算子差异化 bias 构造                             │
│     ├── off_focus tokens（通用压制）                                  │
│     ├── divergent tokens（通用压制）                                  │
│     ├── operator_positive tokens（算子专属·方向引导）                │
│     └── operator_negative tokens（算子专属·方向引导）                │
│                                                                      │
│  7. LLM: generate_with_logit_bias → 文本 + 嵌入向量                  │
│                                                                      │
│  8. EmbeddingObserver.observe(embedding) → 汉字轨迹 + 语义偏离度      │
│     └── set_semantic_deviation() → 影响下一轮的 hybrid_deviation     │
│                                                                      │
│  9. Analyzer: compute_output_deviation(text)                         │
│                                                                      │
│ 10. YinProtocolChecker: check_operator_output()                      │
│                                                                      │
│ 11. judge() → Pass / Rollback / Skip / Stop / FinishCycle            │
│                                                                      │
│ 12. 检查 metabolism_signal → 执行上下文操作                           │
│     ├── Sha → crop history                                           │
│     ├── Zhi → freeze                                                 │
│     └── Cang → clear + restart                                       │
│                                                                      │
│ 13. 周天完成 ? 进入下一轮 : 生成结束                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 六、对幻觉与涌现的处理

### 6.1 项目不试图解决的问题

```
❌ 模型输出的统计数据是否真实
❌ 模型引用的文献是否确实存在
❌ 模型做的推理逻辑在外部事实层面是否正确
```

**三易不知道外部事实。** 它的约束基于形式（卦象、偏离度、格式规则），不基于内容真实性。

### 6.2 项目实际解决的问题

```
✅ 模型是否在正确的算子阶段生成（形式约束）
✅ 模型的输出格式是否符合当前阶段要求（阴仪协议）
✅ 模型是否偏离了用户初始意图（三层注意力协同）
✅ 模型如果偏离，上下文是否被正确裁剪（杀/止/藏）
✅ 系统的每一步内部状态对外部观察者是否可读可审计（白箱化）
```

### 6.3 语义断裂检测（HanziMap 的独特价值）

用 HanziMap 投影做"语义指纹一致性检验"——不是事实检验（"78.3% 对吗？"），
而是语义空间一致性（"模型当前的输出嵌入，和它自己的上一句话，
在汉字投影空间里有没有断裂"）。

如果投影汉字序列突然跳跃（从"分、析、结、果"跳到"神、秘、力、量"），
说明模型发生了**语义断裂**——可能进入了幻觉状态。

这是**不依赖第二个模型的语义断裂检测**，可在当前 8GB 硬件上运行。

### 6.4 路线C 的语义验证（远期）

路线C（太极双LLM）引入独立阴 LLM 做事实校验——这是项目计划中唯一直接对抗幻觉的机制。
前提：双模型硬件条件（16GB+ VRAM）或外部 API。

---

## 七、代码修改清单

### P0 — 观测层接线

| 文件 | 修改 |
|:----|:----|
| `crates/xiang-chat/src/main.rs` | `ConstrainedEngine::new()` 创建 `HanziMap` 并调用 `vm.with_embedding_observer()` |
| `crates/xiang-chat/src/main.rs` | `generate()` 第 315-324 行确认 embedding 返回路径 |

### P1 — 算子差异化 Logit-Bias + 上下文管理

| 文件 | 修改 |
|:----|:----|
| `crates/xiang-llm/src/lib.rs` | `XiangLogitBias` 新增 `operator` 字段 + 算子专属 token 池 + 差异化 bias |
| `crates/xiang-cangvm/src/vm.rs` | 新增 `MetabolismSignal` 枚举 + 杀/止/藏设置信号 |
| `crates/xiang-chat/src/main.rs` | `generate()` 的 judge() 后检查信号并执行上下文操作 |

### P2 — KV cache 管理

| 文件 | 修改 |
|:----|:----|
| `crates/xiang-llm/src/llama_backend.rs` | 实现 `kv_cache_snapshot()` 和 `kv_cache_rollback()` |
| `crates/xiang-llm/src/http_backend.rs` | HttpBackend 回退方案：提示词重打包 |

---

## 八、设计约束

### 8.1 不可违反原则

1. **Logit-Bias 强度上限 ±1.0**——防止过度压制杀死涌现
2. **杀/止/藏只操作已发生的偏离**——不预防未发生的内容
3. **归观测任何时候都接上**——即使 HanziMap 不可用（降级为纯 Hamming）
4. **观测结果不作为惩罚依据**——汉字序列只用于人类可读和白箱审计
5. **三引擎各自可独立测试**——不因集成导致测试失效

### 8.2 与第一性原理的对齐

| 第一性原理要求 | 本设计满足方式 |
|:-------------|:------------|
| 内部状态是人类可读的离散符号 | HanziMap 将连续嵌入投影为汉字序列（观测层） |
| 状态转移是确定性的、可追溯的 | Gua 位运算 + MetabolismSignal 枚举 |
| 思考与表达分离 | 三引擎思考 → LLM 表达（约束下的文本生成） |
| 外部约束是渐进可逼近的 | P0→P1→P2 分阶接线，每步可验证 |
| 底层模型不需要变白箱 | Logit-Bias 方向引导 + 上下文管理，不改权重 |

---

> **观测优先（归+HanziMap）→ 引导其次（生/动/长/育）→ 清理最后（杀/止/藏）。**
>
> **不压制，只引导。不预防，只清理。不假设，只观测。**
