---
name: xiang-hanzi-embed
description: "Hanzi embedding export and validation workflow. Extract token_embd.weight from GGUF model, map to core Hanzi table characters, validate self-similarity and operator centroid accuracy. Triggers: export hanzi embeddings, validate embeddings, hanzi map, embedding extraction, 汉字嵌入导出, 嵌入验证"
---

# 汉字嵌入导出与验证工作流

从 LLM 的 token embedding 矩阵中提取核心字表汉字的嵌入向量，
加载到 HanziMap 并验证映射质量。这是归算子语义观测的基础设施。

## 数据流

```
GGUF 模型文件 (Qwen3.5-4B)
  |  token_embd.weight shape [152064, 2048]
  v
export_hanzi_embeddings.py
  |  对字表每个汉字:
  |    tokenize(ch) -> token_id
  |    token_embd[token_id] -> 2048维向量
  |    L2 normalize
  v
hanzi_embeddings.bin (二进制格式)
  |  [u32 n_chars] [u32 n_embd]
  |  [(u32 char_utf32, u32 token_id, f32 x n_embd) x n_chars]
  v
HanziMap::from_embedded_bytes()
  |  编译时或运行时加载
  v
EmbeddingObserver -> CangVM set_semantic_deviation()
```

---

## Phase 0: 确认字表

### 0.1 字表来源

核心字表在 `crates/xiang-core/src/hanzi_table_data.rs` 中定义，
通过 `include!("hanzi_table_data.rs")` 编译时嵌入。

验证字表完整性:

```bash
cargo test -p xiang-core test_hanzi_table_size
# 要求: Pictogram >= 80, Ideogram >= 80, CompoundIdeogram >= 80
# 要求: 总数 >= 240
```

### 0.2 确认八气算子和八卦名都在字表中

```bash
cargo test -p xiang-core test_eight_operators_present
cargo test -p xiang-core test_eight_trigrams_present
```

---

## Phase 1: 导出嵌入数据

### 1.1 环境准备

```bash
# 确保 llama.cpp 编译支持 embedding 提取
# 需要 Python 3.10+ + llama-cpp-python
pip install llama-cpp-python numpy
```

### 1.2 运行导出脚本

```bash
cd C:\X
python scripts/export_hanzi_embeddings.py \
    --model ./models/qwen3.5-4b-q4_k_m.gguf \
    --output ./crates/xiang-core/data/hanzi_embeddings.bin \
    --n-ctx 1024
```

参数说明:
- `--model`: GGUF 模型文件路径
- `--output`: 输出的二进制嵌入文件路径
- `--n-ctx`: 推理上下文大小（导出用，1024 足够）

### 1.3 验证导出文件

```bash
# 检查文件大小: 约 400 字 x 2048 x 4 bytes = 3.3 MB
ls -la ./crates/xiang-core/data/hanzi_embeddings.bin

# 文件格式验证
python -c "
import numpy as np
data = open('./crates/xiang-core/data/hanzi_embeddings.bin', 'rb').read()
n_chars = int.from_bytes(data[0:4], 'little')
n_embd = int.from_bytes(data[4:8], 'little')
print(f'n_chars={n_chars}, n_embd={n_embd}, size={len(data)} bytes')
# 每个字符: 4(char) + 4(token_id) + n_embd*4(embedding)
assert len(data) == 8 + n_chars * (8 + n_embd * 4)
"
```

---

## Phase 2: 运行时加载

### 2.1 编译时嵌入(推荐)

```rust
// crates/xiang-core/src/hanzi_embeddings.rs 或调用处
const EMBEDDED_DATA: &[u8] = include_bytes!("../data/hanzi_embeddings.bin");
let result = HanziEmbeddings::from_embedded_bytes(EMBEDDED_DATA);
```

### 2.2 运行时加载(调试用)

```rust
// 从文件路径加载（开发调试/动态切换模型）
let embeddings = HanziEmbeddings::load_from_file("path/to/hanzi_embeddings.bin");
```

### 2.3 更新 ConstrainedEngine

```rust
// crates/xiang-chat/src/main.rs -> ConstrainedEngine::new()
// 将当前 HanziMap::empty() 替换为真实加载
let hanzi_emb = HanziEmbeddings::try_load(Some(EMBEDDED_DATA), None)
    .unwrap_or(None)
    .unwrap_or_else(HanziEmbeddings::empty);
let observer = EmbeddingObserver::new(hanzi_emb.map);
vm.embedding_observer = Some(observer);
```

---

## Phase 3: 验证映射质量

### 3.1 自映射验证

每个字表汉字映射到自身时，余弦相似度应接近 1.0:

```rust
#[test]
fn test_self_similarity() {
    let map = load_real_map();
    for ch in CORE_CHARS {
        let emb = map.embedding_of(ch).unwrap();
        let result = map.map_single(emb).unwrap();
        assert_eq!(result.ch, ch, "{}: self-mapping failed", ch);
        assert!(result.similarity > 0.95, "{}: low self-sim {}", ch, result.similarity);
    }
}
```

### 3.2 算子语义锚点验证

验证四个算子(生/动/长/育)的 centroid 分类准确率:

```rust
#[test]
fn test_operator_centroid_with_real_embeddings() {
    let map = load_real_map();
    let anchors = build_operator_anchors(&map);
    assert_eq!(anchors.len(), 4);

    // 验证每个算子的锚定汉字被正确分类到自己的算子
    for (op_ch, anchor_chars) in &OPERATOR_CHARS {
        let centroid = anchors.iter()
            .find(|a| a.operator == *op_ch)
            .unwrap();
        for ch in anchor_chars {
            if let Some(emb) = map.embedding_of(*ch) {
                let result = classify_operator_phase(emb, &anchors, 0.1);
                // 期望: 这个锚定汉字被分到它所属的算子
                assert_eq!(result, Some(*op_ch),
                    "char {} should classify as {}, got {:?}", ch, op_ch, result);
            }
        }
    }
}
```

### 3.3 近义词区分验证

验证语义相近的字在嵌入空间中确实相近:

```rust
#[test]
fn test_semantic_proximity() {
    let map = load_real_map();
    // 水 and 火 are both elements -> should be closer than 水 and 飞
    let water = map.embedding_of('水').unwrap();
    let fire = map.embedding_of('火').unwrap();
    let fly = map.embedding_of('飞').unwrap();

    let water_fire = cosine_similarity(water, fire);
    let water_fly = cosine_similarity(water, fly);
    assert!(water_fire > water_fly,
        "water-fire sim ({}) should > water-fly sim ({})", water_fire, water_fly);
}
```

### 3.4 运行所有验证

```bash
cargo test -p xiang-core test_hanzi_self_similarity
cargo test -p xiang-core test_operator_centroid
cargo test -p xiang-core test_semantic_proximity
```

---

## Phase 4: CangVM 集成验证

### 4.1 EmbeddingObserver 注入验证

确认 CangVM 的 embedding_observer 字段非空:

```rust
#[test]
fn test_observer_injected() {
    let map = load_real_map();
    let observer = EmbeddingObserver::new(map);
    let mut vm = CangVM::new();
    vm.embedding_observer = Some(observer);
    assert!(vm.embedding_observer.is_some(), "Observer should be injected");
}
```

### 4.2 观测流程验证

```rust
#[test]
fn test_observe_returns_some_with_no_empty_map() {
    let map = load_real_map();
    let mut observer = EmbeddingObserver::new(map);
    assert!(observer.is_ready(), "Map should be loaded");

    // 用嵌入向量模拟 LLM 输出
    let emb = vec![0.5; 2048];  // 须与真实 n_embd 匹配
    let obs = observer.observe(&emb);
    assert!(obs.is_some(), "observe() should return Some with loaded map");
    // semantic_deviation 应在 [0, 1] 区间
    let dev = observer.semantic_deviation();
    assert!(dev >= 0.0 && dev <= 1.0, "deviation out of range: {}", dev);
}
```

### 4.3 集成测试

```bash
cargo test -p xiang-core -p xiang-cangvm
```

---

## Phase 5: Chat 管道验证

### 5.1 编译确认

```bash
cargo check -p xiang-core -p xiang-chat
```

### 5.2 运行时观测

启动 chat 服务后，观察控制台输出确认汉字轨迹打印:

```
[归.观测] 锚点已设置: 维度=2048
[归.观测] 汉字轨迹(最近5): 探,索,分,析,方 | 语义偏离: 0.123 | 弱锚定比: 0.00
```

如果出现 `weak_anchor_ratio` 持续偏高(> 0.5)，说明字表嵌入质量不够，
需要检查导出脚本或增加字表覆盖。

---

## 故障处理

| 问题 | 可能原因 | 解决 |
|:----|:--------|:----|
| embedding 自映射相似度 < 0.9 | L2 归一化缺失 | 检查 export 脚本的 normalize 逻辑 |
| 算子 centroid 分类准确率 < 60% | 锚定汉字嵌入在 tokenizer 中被拆为 subword | 检查 tokenize 结果: 确认每个字单独成 token |
| binary 文件大小异常 | n_embd 维度不匹配 | 运行时 llama_n_embd() 动态获取 |
| HanziMap::from_embedded_bytes 返回 Err | 二进制格式不匹配 | 确认字节序(little-endian)和字段对齐 |
| EmbeddingObserver 全部 None | n_embd 维度与 LLM 实际不匹配 | print(n_embd) 确认 |
