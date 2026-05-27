# 归藏生成树闭环 — 实施计划

> 本文档是对 [specs/15-guisheng-chengshu-bi.md](../specs/15-guisheng-chengshu-bi.md) 设计文档的执行计划。
> 所有代码修改以该文档为锚点。

## 修改时序

```
Phase 1 (P0) — 观测层接线
  └── xiang-chat/main.rs: 注入 EmbeddingObserver

Phase 2 (P1) — 干预层升级 + 管理层实现
  ├── xiang-llm/lib.rs: 算子差异化 Logit-Bias
  ├── xiang-cangvm/vm.rs: MetabolismSignal
  └── xiang-chat/main.rs: 上下文操作

Phase 3 (P2) — KV cache 管理
  ├── xiang-llm/llama_backend.rs: seq_id 快照/回滚
  └── xiang-llm/http_backend.rs: 提示词重打包
```

## 验证标准

| 阶段 | 验证 |
|:----|:----|
| P0 | `cargo test` 通过 + EmbeddingObserver 被注入（非 None） |
| P1 | 算子差异化 bias 生效（MockBackend 可验证）+ 上下文裁剪正确 |
| P2 | KV cache snapshot/rollback 返回非 -1 值 |
