# 象（Xiang）项目 — AI 助手行为规则

> 版本：1.0 | 最后更新：2026-05-27
>
> 本文档定义了 AI 助手在本项目中的所有行为准则。所有交互、注释、文档必须使用中文。

---

## 一、项目身份标识

**项目名称**：象（Xiang）— 三易智能约束系统

**核心定位**：基于形式化认知约束语言（象语言）的推理时模型控制系统。不训练模型变聪明，而是用确定性状态机，为概率性的 LLM 生成过程装上方向盘、油门和刹车。

**技术栈**：

| 层面 | 技术 | 说明 |
|------|------|------|
| 后端 | Rust 2024 edition | 9 个工作区 crate |
| Web 服务 | axum + tokio | xiang-chat crate |
| 前端 | React 18 + TypeScript | chat-ui 目录 |
| 构建 | Vite 6 | 前端构建工具 |
| 样式 | Tailwind CSS 3.4 | 前端样式 |
| LLM 推理 | llama.cpp (Vulkan) | GGUF 模型推理 |
| LLM 后端 | 双后端：llama_backend / http_backend | 本地/远程 |

**9 个 crate 职责**：

| Crate | 职责 | 测试数 |
|-------|------|:------:|
| `xiang-core` | 核心类型：Gua/Bagua/偏离度/CangSea/阴仪协议/汉字表/嵌入 | ~77 |
| `xiang-parser` | 象语言词法/语法/AST 解析器 | ~12 |
| `xiang-cangvm` | 归藏执行引擎 + 三引擎编排 + 阶段约束决策 | ~60 |
| `xiang-shanvm` | 连山 6 步策略决策链 | ~17 |
| `xiang-zhouvm` | 周易 8 卦网格 + Markov 温度调度 | ~14 |
| `xiang-llm` | Logit-bias 生成 + LLM 后端抽象 | ~16 |
| `xiang-verifier` | 可信验证 | ~14 |
| `xiang-experiments` | 实验框架 + 基准测试（CLI binary） | ~33 |
| `xiang-chat` | Axum 聊天服务 + API 层 | ~2 |

---

## 二、语言规则（硬性约束）

### 2.1 沟通语言

- **所有 AI 回复必须使用简体中文**。包括解释、分析、建议、报告等。
- 术语保留：代码中的英文标识符（`Gua`、`CangVM`、`Bagua` 等）在解释时可保留原名。
- 错误消息中出现的 Rust 编译错误原文保留英文，但解释用中文。

### 2.2 代码注释

- 所有 Rust 文档注释（`///`）使用中文。
- 所有 Rust 普通注释（`//`、`/* */`）使用中文。
- 所有 TypeScript/React 组件注释使用中文。
- 所有 Markdown 文档使用中文。
- 代码内的变量名、函数名、类型名保持英文（Rust 惯例），**不**改为拼音或中文。

### 2.3 文档与规范

- 所有 spec 文件、README、ARCHITECTURE.md 等使用中文。
- 新增文档必须使用中文撰写。
- 中文文档中技术术语首次出现时，可附英文原名，例如：卦象状态向量（Gua）。

---

## 三、代码风格规范

### 3.1 Rust 代码规范

#### 3.1.1 格式

- 使用 Rust 2024 edition 的默认格式化风格（`cargo fmt`）。
- 缩进：4 空格。
- 行宽：100 字符以内。
- 导入分组：`std` → 外部依赖 → `crate` → `super`/`self`，组间空行分隔。

#### 3.1.2 命名

| 项 | 规范 | 示例 |
|----|------|------|
| 类型/枚举 | PascalCase | `CangVM`, `Bagua`, `FangWei` |
| 函数/方法 | snake_case | `execute_qi_op`, `current_deviation` |
| 变量 | snake_case | `deviation_log`, `obstruction_count` |
| 常量 | SCREAMING_SNAKE_CASE | `MAX_ENTRIES`, `SAN_YI_CONSTRAINT_FULL_TEXT` |
| 模块 | snake_case | `yin_checker`, `cang_sea` |
| 枚举变体 | PascalCase | `CangVMDecision::Pass`, `QiOp::Sheng` |

#### 3.1.3 文档注释

- 所有 `pub` 项（类型、函数、字段、枚举变体）必须有 `///` 文档注释。
- 文档注释结构：第一行简要说明 → 空行 → 详细说明 → 空行 → `# 示例`（可选）。
- 模块级注释用 `//!`。
- 重要算法必须包含形式化描述，如偏离度公式、状态变换规则。

```rust
/// 卦 (Gua) — 6 位状态向量 S ∈ {0..63}
///
/// 映射到归藏易 64 卦。
/// 不变量：self.0 ∈ [0, 63]。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Gua(pub u8);
```

#### 3.1.4 错误处理

- 使用自定义错误枚举（`VmError`），实现 `Display` 和 `Error` trait。
- 错误消息用中文。
- 不使用 `unwrap()`、`expect()`、`panic!()` 在生产代码中——只在测试和 `from_u8` 等显式安全断言中使用。
- 使用 `Result<_, E>` 返回可恢复错误。

```rust
#[derive(Debug, Clone)]
pub enum VmError {
    UndefinedVariable(String),
    InvalidState(String),
}
impl std::fmt::Display for VmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VmError::UndefinedVariable(s) => write!(f, "未定义变量: {s}"),
            VmError::InvalidState(s) => write!(f, "无效状态: {s}"),
        }
    }
}
```

#### 3.1.5 模块组织

- 每个 crate 的 `src/lib.rs` 只做模块声明和 `pub use` 重导出。
- 功能代码放在对应的模块文件中。
- 模块文件不超过 800 行。超过则拆分子模块。

```
xiang-core/src/
├── lib.rs              # 模块声明 + pub use 重导出
├── gua.rs              # Gua 6 位状态向量
├── bagua.rs            # 八卦 8 状态
├── cang_sea.rs         # 藏海 Hebbian 矩阵 + 语义存储
├── deviation.rs        # 偏离度计算
├── yin_checker.rs      # 阴仪协议校验
├── embedding.rs        # 嵌入向量
├── semantic.rs         # 语义决策/姿态
├── hanzi_table.rs      # 汉字表
├── hanzi_map.rs        # 汉字映射
└── ...
```

### 3.2 测试规范

#### 3.2.1 测试位置

- 单元测试：内联在源文件末尾的 `#[cfg(test)] mod tests { ... }` 中。
- 集成测试：放在 crate 根目录的 `tests/` 目录下。目前多数 crate 的 `tests/` 为空，集成测试可逐步补充。
- `xiang-cangvm` 的测试包含三层：单元测试（`cangvm::tests`）、周天循环测试（`zhou_tian_cycle_tests`）、阶段约束测试（`phase_constraint_tests`）。

#### 3.2.2 测试命名

- 使用蛇形命名，如 `test_execute_sheng`、`test_deviation_full`。
- 命名模板：`test_<被测试功能>_<场景/条件>_<期望结果>`。

#### 3.2.3 测试覆盖要求

- 所有 `pub fn` 必须有至少一个测试。
- 所有 8 个算子的位运算必须对所有 64 个 Gua 状态测试不变量（见 `test_all_ops_stay_in_range`）。
- 偏离度计算必须测试边界值（0.0, 1.0, 0.5）。
- 阶段约束的 judge() 决策必须测试所有分支：Pass / Rollback / Skip / Stop / FinishCycle。
- 连山 6 步决策链必须测试所有 7 种方位输出至少一次。

#### 3.2.4 测试结构

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_expected_behavior() {
        // Arrange
        let mut vm = CangVM::new();

        // Act
        vm.execute(&stmts).unwrap();

        // Assert
        assert_eq!(vm.state, expected);
        assert!(vm.cang_sea.is_empty() == expected_empty);
    }
}
```

### 3.3 前端规范（TypeScript + React）

#### 3.3.1 格式

- 使用项目已有的配置：`tsconfig.app.json` + 路径别名 `@/`。
- 缩进：2 空格。
- 使用函数组件 + Hooks，不使用 class 组件。
- 使用 TypeScript 严格模式。

#### 3.3.2 命名

| 项 | 规范 | 示例 |
|----|------|------|
| 组件 | PascalCase | `ChatPanel`, `StatusBar` |
| 函数 | camelCase | `sendMessage`, `fetchState` |
| 变量 | camelCase | `rawMessages`, `constrainedMeta` |
| 类型/接口 | PascalCase | `Message`, `ConstrainedMeta` |
| 文件 | PascalCase（组件）、camelCase（工具） | `ChatPanel.tsx`, `utils.ts` |

#### 3.3.3 样式

- 使用 Tailwind CSS 工具类，不写独立 CSS 文件（`index.css` 只放基础主题）。
- 组件内使用 `cn()` 工具函数（`clsx` + `tailwind-merge`）合并类名。

#### 3.3.4 前后端交互

- 通过 Vite proxy（`/api` → `localhost:3001`）转发请求。
- API 路径规范：
  - `GET /api/state` — 获取状态
  - `POST /api/raw` — 无约束生成
  - `POST /api/constrained` — 带约束生成
  - `POST /api/reset` — 重置会话

---

## 四、架构规则

### 4.1 三引擎架构

**核心原则**：三引擎（归藏/连山/周易）各自独立可测试，通过 CangVM 统一编排。

```
用户请求 → 周易选姿态 → 归藏执行周天 → 遇障碍 → 连山导航 → 归藏执行策略
                                                              ↓
                                                        藏海沉淀经验
                                                              ↓
                                                        经验回流各引擎
```

- **归藏（CangVM）**是中心执行引擎，包含阶段约束决策逻辑（`judge()`）。
- **连山（ShanVM）**是纯函数式策略引擎，输入当前状态输出方位决策，不持有可变状态。
- **周易（ZhouVM）**是认知姿态引擎，管理 8 卦 Markov 链和温度调度。
- **藏海（CangSea）**是经验存储，被归藏写入，被连山/周易读取。

### 4.2 依赖方向

```
xiang-chat
  ├── xiang-cangvm
  │    ├── xiang-core
  │    ├── xiang-parser
  │    ├── xiang-shanvm
  │    └── xiang-zhouvm
  ├── xiang-llm
  ├── xiang-shanvm
  └── xiang-core
```

**依赖规则**：上层可依赖下层，下层不可依赖上层。
- `xiang-core` 是底层基础，不依赖任何其他 crate。
- `xiang-shanvm`、`xiang-zhouvm` 只依赖 `xiang-core`。
- `xiang-cangvm` 依赖所有下层并编排三引擎。

### 4.3 LLM 后端抽象

- `xiang-llm` 提供 `XiangLogitBias`（生成前 token 概率调整）和 LLM 后端抽象。
- 双后端：`llama_backend`（本地 FFI）和 `http_backend`（远程 API）。
- 通过 Cargo features 切换：`default = []`，`llama_backend` 或 `http_backend`。

### 4.4 三路线设计

| 路线 | 方式 | 状态 |
|:----:|------|:----:|
| 路线A | 外部程序化约束（已实现） | ✅ 稳定运行 |
| 路线B | LoRA 微调内化约束原语（可选方案） | 📋 硬件阻塞 |
| 路线C | 双 LLM + 全语义化（远期愿景） | 🔮 远期规划 |

**增删改代码前的决策树**：
1. 改路线A 代码？→ 确保不影响 `YinProtocolChecker`、`XiangLogitBias`、`CangVM::judge()` 的现有行为。
2. 接连山/周易入流水线？→ 在 `run_phase_cycle()` 中扩展三引擎协同部分。
3. 改路线B/C 规范？→ 只修改 `specs/` 目录下的 MD 文件，不改 Rust 代码。

---

## 五、工作流程规则

### 5.1 文件修改前

- 修改 Rust 代码前先 `cargo check` 确认当前编译状态。
- 修改前端代码前确认 `npm run build` 能通过。
- 修改跨 crate 接口时，先理解依赖关系和类型定义。

### 5.2 修改后验证

- Rust 代码修改后必须：
  1. `cargo check` — 无编译错误
  2. 受影响的 crate 的测试全部通过
  3. `cargo clippy` — 无新增 warning（优先处理 lint）
- 前端代码修改后必须：
  1. `tsc --noEmit` — 无类型错误
  2. 手动测试 UI 关键路径

### 5.3 测试运行

```bash
# 运行所有测试
cargo test

# 运行单个 crate 测试
cargo test -p xiang-core

# 运行单个测试
cargo test -p xiang-cangvm test_judge_pass_advances_operator

# 运行基准测试（Mock 模式）
cargo run --bin xiang-exp -- bench --mock offfocus --seed 42

# 运行基准测试（真实模型）
cargo run --bin xiang-exp --features http_backend -- bench --http http://localhost:8080
```

### 5.4 新增代码流程

1. 确定属于哪个 crate。
2. 遵循已有模式——同文件中找最接近的已有函数，模仿其结构和风格。
3. 添加 `pub` 的文档注释（中文）。
4. 添加内联测试。
5. `cargo check && cargo test` 验证。
6. 如果变更跨 crate（如修改 `xiang-core` 的类型定义），更新所有依赖 crate 的相应代码。

### 5.5 不可违反的原则

- ❌ **不使用 `as any` / `@ts-ignore` / `@ts-expect-error`** 绕过类型系统。
- ❌ **不使用空 catch 块** `catch(e) {}`。
- ❌ **不删除已有测试为了"让构建通过"**。
- ❌ **不猜测未读取的代码**——先读文件再修改。
- ❌ **不随意新增依赖**——优先使用项目已有依赖。
- ❌ **Bug 修复不附带重构**——最小改动原则。
- ❌ **route B 代码实现不带入 route A**——两者通过 features 隔离。

---

## 六、沟通规则

### 6.1 回复风格

- **直接、无前缀**：不要以"好的"、"让我"、"明白了"开头，直接给答案或开始工作。
- **使用 todo 列表**：多步骤任务立即创建 `todowrite`。
- **报告测试结果**：如实报告——测试失败就贴输出，没跑就说没跑。

### 6.2 信息呈现

- 使用 Markdown 表格组织结构化信息。
- 代码块标注语言（`rust`、`typescript`、`bash`、`text`）。
- 文件路径使用超链接格式：`[文件名](file:///C:/X/path/to/file.rs)`。

### 6.3 建议与质疑

- 发现设计问题 → 指出问题 + 给出替代方案 + 询问意向。
- 不确定意图 → 确认理解后再行动，不猜测实现。

---

## 七、关键文件索引

| 文件 | 内容 | 重要性 |
|------|------|:------:|
| `Cargo.toml` | 工作区定义、9 个 crate 成员 | ⭐⭐⭐ |
| `ARCHITECTURE.md` | 系统架构、组件依赖、技术栈 | ⭐⭐⭐ |
| `specs/00-总纲.md` | 规范导航、三引擎状态、路线图 | ⭐⭐⭐ |
| `specs/01-路线A-阶段约束.md` | 归藏引擎实现规格 | ⭐⭐⭐ |
| `specs/04-象语言规范.md` | 象语言语法规范 | ⭐⭐ |
| `specs/05-三易算法内核.md` | 算法层完整定义 | ⭐⭐⭐ |
| `crates/xiang-core/src/lib.rs` | 核心类型重导出入口 | ⭐⭐⭐ |
| `crates/xiang-cangvm/src/vm.rs` | 主执行引擎 | ⭐⭐⭐ |
| `crates/xiang-core/src/gua.rs` | 6 位卦象 + 八气算子 | ⭐⭐⭐ |
| `crates/xiang-core/src/cang_sea.rs` | 藏海 Hebbian 矩阵 | ⭐⭐ |
| `crates/xiang-core/src/yin_checker.rs` | 阴仪协议校验 | ⭐⭐ |
| `crates/xiang-shanvm/src/lib.rs` | 连山策略引擎 | ⭐⭐ |
| `crates/xiang-zhouvm/src/lib.rs` | 周易认知姿态引擎 | ⭐⭐ |
| `crates/xiang-llm/src/lib.rs` | Logit-bias + LLM 抽象 | ⭐⭐ |
| `crates/xiang-experiments/src/main.rs` | 基准测试 CLI | ⭐ |
| `crates/xiang-chat/src/main.rs` | Axum 聊天服务 | ⭐⭐ |
| `chat-ui/src/App.tsx` | 前端主界面 | ⭐⭐ |
| `chat-ui/src/components/ChatPanel.tsx` | 聊天面板组件 | ⭐ |

---

> **核心准则：凡涉及修改，先读相关代码，遵循既有模式，最小改动，验证通过。**
>
> **三引擎排序：归藏是执行者，连山是导航者，周易是姿态者。三者合一，构成完整的认知约束闭环。**
