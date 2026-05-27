---
name: xiang-wiring
description: "Three-engine signal wiring workflow. Involves signal flow between CangVM/ShanVM/ZhouVM, xiang-core type definitions, xiang-llm LogitBias extensions, xiang-chat ConstrainedEngine consumption. Triggers: signal wiring, new signal channel, three-engine collaboration, signal flow"
---

# 三引擎信号线接线工作流

定义在归藏/连山/周易三个引擎之间建立新信号通道的完整流程。
每次接线涉及 3-4 个 crate、多个文件，容易遗漏。本技能提供标准操作流程。

## 架构映像

```
CangVM (xiang-cangvm)
  |  产生信号
  |  +-- metabolism_signal (沙/止/藏的操作信号)
  |  +-- semantic_deviation (语义偏离度)
  |  +-- [新信号]
  v
xiang-core 类型定义
  |  定义信号类型、枚举、trait
  v
xiang-llm Logit-Bias / 后端
  |  消费信号 -> token 级偏置
  v
xiang-chat ConstrainedEngine
  |  消费信号 -> 上下文操作 / 对话管理
  v
验证: 单元测试 + 集成测试
```

**核心原则**: CangVM 只产生信号，不执行操作。执行由下游组件负责。

---

## Phase 0: 信号设计

### 0.1 确定信号归属

信号属于哪个层?

| 层 | 定位 | 产生者 | 消费者 |
|:---|:-----|:------|:------|
| 观测层 | 纯观测，不修改状态 | 归算子 + EmbeddingObserver | CangVM judge() / ZhouVM |
| 干预层 | 生成时 token 级方向引导 | CangVM 当前算子阶段 | XiangLogitBias |
| 管理层 | 生成后上下文操作 | 沙/止/藏 算子 | ConstrainedEngine |

### 0.2 定义信号类型

```rust
// xiang-cangvm/src/vm.rs:
// 枚举变体 -> MetabolismSignal(管理层)
// 新字段 -> CangVM 结构体
pub struct CangVM {
    // ... 现有字段 ...
    // [+] pub new_signal: Option<NewSignalType>,
}
```

### 0.3 确认信号流向

```
谁产生? --> 谁消费? --> 谁执行?
CangVM 算子 -> 信号 -> XiangLogitBias -> token 概率改变
CangVM 算子 -> 信号 -> ConstrainedEngine -> 对话历史改变
EmbeddingObserver -> 信号 -> CangVM judge() -> 决策改变
```

---

## Phase 1: CangVM 信号产生

### 1.1 字段到 CangVM 结构体

```rust
// crates/xiang-cangvm/src/vm.rs
pub struct CangVM {
    /// [+] 文档注释: 用途、触发条件、消费者
    pub new_field: Option<NewType>,
}
```

### 1.2 new() 中初始化

```rust
new_field: None,
```

### 1.3 在算子里设信号

```rust
// execute_phase_operator() 中
"[算子名]" => {
    // ... 现有位运算 ...
    self.new_field = Some(value);
    Ok(())
}
```

### 1.4 导出到 lib.rs

```rust
// crates/xiang-cangvm/src/lib.rs
pub use vm::{CangVM, ..., /* [+] 新类型 */};
```

### 1.5 验证

```bash
cargo test -p xiang-cangvm
```

---

## Phase 2: xiang-core 类型定义

如果新信号需要在 crate 之间共享类型:

```rust
// crates/xiang-core/src/lib.rs
pub mod new_module;
pub use new_module::{...};
```

验证:

```bash
cargo test -p xiang-core
```

---

## Phase 3: xiang-llm 消费

### 3.1 扩展 XiangLogitBias

```rust
// crates/xiang-llm/src/lib.rs
pub struct XiangLogitBias {
    // [+] pub new_field: Option<Vec<u32>>,
}

// bias_for_step() 中使用
// 强度上限 +-1.0
```

### 3.2 扩展 generate_constrained_turn

```rust
pub fn generate_constrained_turn(
    // ... 现有参数 ...
    // [+] new_param: NewType,
) -> Result<GenerationResult, LlmError> {
    let bias = XiangLogitBias::new(...)
        // [+] .with_new_param(...)
        .with_strategy(strategy_bias);
}
```

### 3.3 更新 box_clone

```rust
fn box_clone(&self) -> Box<dyn LogitBias + Send> {
    Box::new(XiangLogitBias {
        // [+] new_field: self.new_field.clone(),
    })
}
```

### 3.4 验证

```bash
cargo test -p xiang-llm
```

---

## Phase 4: xiang-chat ConstrainedEngine 消费

### 4.1 传递参数

```rust
// crates/xiang-chat/src/main.rs
let result = self.ctx.generate_constrained_turn(
    // ... 现有参数 ...
    // [+] new_param_value,
)?;
```

### 4.2 后处理信号

```rust
// generate() 中 judge() 之后
if let Some(signal) = &self.cang_vm.new_field {
    self.handle_new_signal(signal);
}
```

### 4.3 实现信号处理方法

```rust
impl ConstrainedEngine {
    fn handle_new_signal(&mut self, signal: &NewType) {
        // 实现上下文操作
    }
}
```

### 4.4 更新 ConstrainedResponse(可选)

如果前端需要显示新信号:

```rust
pub struct ConstrainedResponse {
    // [+] pub new_field: Option<NewType>,
}
```

### 4.5 更新 import

```rust
use xiang_cangvm::{CangVM, MetabolismSignal /* [+] 新类型 */};
use xiang_core::{/* [+] 新类型 */};
```

### 4.6 验证

```bash
cargo check -p xiang-chat
cargo test -p xiang-chat
```

---

## Phase 5: 验证与回归

### 5.1 编译验证

```bash
cargo check
```

### 5.2 单元测试

```bash
cargo test -p xiang-core -p xiang-cangvm -p xiang-llm -p xiang-chat
```

### 5.3 Lint

```bash
cargo clippy -p xiang-cangvm -p xiang-llm -p xiang-chat
```

### 5.4 Mock 集成测试(可选)

```rust
#[test]
fn test_new_signal_flow() {
    // MockBackend + CangVM
    // 设触发条件
    // 验证信号被消费
}
```

---

## 反模式

| 违反 | 后果 | 严重度 |
|:----|:-----|:------|
| CangVM 直接操作对话历史或 KV cache | 违反只发信号原则，耦合 | CRITICAL |
| 信号类型不导出 | 下游无法引用 | CRITICAL |
| 新信号 new() 未初始化 None | 未定义行为 | CRITICAL |
| box_clone() 漏新字段 | 运行时 panic | HIGH |
| bias 强度超过 +-1.0 | 杀死涌现 | HIGH |
| 忘更新 experiments 的调用签名 | 编译失败 | MEDIUM |
| 新信号无中文文档注释 | 违反项目规则 | MEDIUM |
