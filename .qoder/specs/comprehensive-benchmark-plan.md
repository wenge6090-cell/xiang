# 象 (Xiang) — 全面基准测试实施计划

## Context

象项目是一个Rust工作空间，实现了"三易智能约束系统"——三个协同的虚拟机（归藏CangVM、连山ShanVM、周易ZhouVM）通过确定性约束提升LLM生成质量。

**当前问题**:
1. ShanVM和ZhouVM代码已就绪但实验框架未注入三引擎
2. 没有正式的基准测试框架，无法对比"裸跑 vs 三易控制"
3. 缺少统计显著性分析（t-test/p-value/effect size）
4. 需要100组数据证明"三易控制能显著提升本地模型能力"

**目标**: 用一个任务、一个模型、两个条件（裸跑 vs 三易控制），跑出100组数据，算出偏离率、结构合规率、完成率的统计显著性差异。

---

## 关键洞察

- `xiang-experiments` 已依赖 `xiang-shanvm` 和 `xiang-zhouvm`（Cargo.toml已配置）
- `CangVM` 已有 `with_shan_vm()` / `with_zhou_vm()` builder方法
- `CangVM` 已有 `run_shan_vm()` / `run_zhou_vm_step()` 执行方法
- 周天循环中已有 `dev > 0.5` 触发ShanVM的逻辑
- 唯一缺口：`ExperimentRunner::run_experiment()` 调用 `CangVM::new()` 时未注入ShanVM/ZhouVM
- MockBackend的`OffFocus`模式产生~30%偏离率，适合对比测试

---

## 实施计划

### Phase 1: 三引擎接入（修改 lib.rs）

**文件**: `crates/xiang-experiments/src/lib.rs`

1. **`ExperimentConfig` 新增字段**:
   - `pub enable_three_engine: bool` — 控制是否注入ShanVM/ZhouVM（默认true）

2. **`ExperimentMetrics` 新增字段**:
   - `pub completion_rate: f32` — 完成率
   - `pub structure_compliance_rate: f32` — 结构合规率

3. **修改 `run_experiment()`**:
   - 在 `let mut vm = CangVM::new();` 之后，当 `use_guidance && config.enable_three_engine` 时：
     ```rust
     vm = vm.with_shan_vm(ShanVM::new()).with_zhou_vm(ZhouVM::new());
     ```
   - 在指标收集阶段计算 `completion_rate` 和 `structure_compliance_rate`
   - 在phase_constraint_mode路径也注入三引擎

4. **所有预定义实验的 `ExperimentConfig`** 添加 `enable_three_engine: true`

### Phase 2: 统计模块（新建 statistics.rs）

**文件**: `crates/xiang-experiments/src/statistics.rs`

零外部依赖实现（~150行）：

```rust
pub struct MetricStats { mean, std_dev, min, max, median, n }
pub struct TTestResult { t_statistic, p_value, cohens_d, significant, interpretation }

pub fn compute_metric_stats(values: &[f64]) -> MetricStats
pub fn welch_t_test(a: &[f64], b: &[f64]) -> TTestResult  // Welch-Satterthwaite
pub fn cohens_d(a: &[f64], b: &[f64]) -> f64
```

算法：
- Welch's t: `t = (mean_a - mean_b) / sqrt(var_a/n_a + var_b/n_b)`
- 自由度: Welch-Satterthwaite 近似
- p-value: 用 Student t CDF 近似（误差函数 + Beta不完全函数）
- Cohen's d: `|mean_a - mean_b| / sqrt((var_a + var_b) / 2)`

### Phase 3: 输入生成（新建 inputs.rs）

**文件**: `crates/xiang-experiments/src/inputs.rs`

10个主题 × 10个变体 = 100组不同输入：
- 主题: AI认知架构, 机器学习基础, 自然语言处理, 分布式系统, 数据库设计, 软件架构, 网络安全, 算法优化, 前端开发, 项目管理
- 每主题10个变体，覆盖6种认知任务：解释、比较、分析、设计、评估、调试

```rust
pub fn generate_benchmark_inputs(seed: u64) -> Vec<String>
```

### Phase 4: 基准运行器（新建 benchmark.rs）

**文件**: `crates/xiang-experiments/src/benchmark.rs`

核心结构：
```rust
pub struct BenchmarkConfig { num_trials: usize, inputs: Vec<String>, base_config: ExperimentConfig }
pub struct BenchmarkTrial { trial_index, input, controlled: ExperimentMetrics, constrained: ExperimentMetrics }
pub struct BenchmarkResults { num_trials, trials: Vec<BenchmarkTrial>, summary: BenchmarkSummary }
```

`BenchmarkRunner::run()` 循环100次：
- 每次创建全新CangVM（对照组裸跑 / 实验组三引擎）
- 收集偏离率、结构合规率、完成率
- 调用 `statistics::compute_summary()` 计算统计显著性

### Phase 5: 增强报告（新建 report.rs）

**文件**: `crates/xiang-experiments/src/report.rs`

```rust
pub fn generate_benchmark_html_report(results: &BenchmarkResults) -> String
```

HTML内容：
1. 汇总表（mean±std | t | p-value | Cohen's d | 显著标记）
2. CSS柱状图对比三指标
3. 100组逐对散点表
4. 自动结论：基于p-value判断显著性

### Phase 6: CLI扩展（修改 main.rs）

**文件**: `crates/xiang-experiments/src/main.rs`

新增 `bench` 子命令：
```
xiang-exp bench                    # 100次基准测试(MockBackend)
xiang-exp bench --model <GGUF>     # 真实模型
xiang-exp bench --http <URL>       # 远程llama.cpp
```

输出：`benchmark_report.html` + 控制台关键统计摘要

---

## 验证计划

1. **单元测试**: 每个新模块都有测试覆盖
   - statistics: 用已知数据集验证t-test（与Python scipy.stats对比）
   - inputs: 验证生成100个不同非空输入
   - benchmark: 小规模验证（3-5次试验）

2. **集成测试**: `cargo test -p xiang-experiments` 全部通过

3. **端到端验证**: `cargo run -p xiang-experiments -- bench`
   - 输出包含统计显著性结论
   - p-value < 0.05 才算显著

---

## 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `crates/xiang-experiments/src/lib.rs` | **修改** | 三引擎注入 + Config/Metrics扩展 |
| `crates/xiang-experiments/src/statistics.rs` | **新建** | Welch's t-test统计模块 |
| `crates/xiang-experiments/src/inputs.rs` | **新建** | 100组输入生成器 |
| `crates/xiang-experiments/src/benchmark.rs` | **新建** | 基准运行器核心 |
| `crates/xiang-experiments/src/report.rs` | **新建** | 统计增强HTML报告 |
| `crates/xiang-experiments/src/main.rs` | **修改** | 新增bench子命令 |
