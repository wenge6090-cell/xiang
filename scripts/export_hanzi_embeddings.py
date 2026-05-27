#!/usr/bin/env python3
"""
汉字嵌入向量导出脚本 (v2)

使用 llama-tokenize.exe 获取准确的 token ID，
然后用 gguf 库直接从 GGUF 中提取 token embedding 矩阵。

策略:
  1. 从 Rust 源码提取汉字列表
  2. 用 llama-tokenize.exe 逐字分词获取 token ID(s)
  3. 用 gguf 库读取 token_embd.weight tensor
  4. 对多 token 汉字取平均嵌入
  5. 输出 HanziMap 兼容的二进制格式

用法:
  python export_hanzi_embeddings.py \
    --model models/qwen2.5-0.5b-instruct-f16.gguf \
    --output data/hanzi_embeddings.bin \
    [--tokenize-tool llama.cpp/build/bin/Release/llama-tokenize.exe]
"""

import argparse
import os
import re
import struct
import subprocess
import sys
from pathlib import Path


# ─── 字符提取 ───────────────────────────────────────────────────

def extract_chars_from_rust_source(source_path: str) -> list[str]:
    with open(source_path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = r"HanziEntry::[pic]\('(.)'"
    chars = list(dict.fromkeys(re.findall(pattern, content)))
    print(f"[INFO] 从 {source_path} 提取到 {len(chars)} 个唯一汉字")
    return chars


# ─── 分词 (llama-tokenize) ──────────────────────────────────────

def tokenize_chars(tokenize_tool: str, model_path: str, chars: list[str]) -> dict[str, list[int]]:
    """
    逐个汉字调用 llama-tokenize 获取 token ID 列表。
    返回: {char: [token_id, ...]}
    """
    char_to_tokens: dict[str, list[int]] = {}
    total = len(chars)

    for idx, ch in enumerate(chars):
        result = subprocess.run(
            [tokenize_tool, "--model", model_path, "--no-bos", "--prompt", ch],
            capture_output=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
        output = result.stderr + result.stdout
        token_ids = []
        for line in output.split("\n"):
            stripped = line.strip()
            if "->" in stripped:
                parts = stripped.split()
                if parts and parts[0].isdigit():
                    token_ids.append(int(parts[0]))

        if token_ids:
            char_to_tokens[ch] = token_ids
        else:
            print(f"[WARN] 无法分词: '{ch}' (U+{ord(ch):04X})")

        if (idx + 1) % 50 == 0 or idx == total - 1:
            print(f"[PROGRESS] 分词 {idx + 1}/{total}")

    return char_to_tokens


# ─── 嵌入矩阵提取 ───────────────────────────────────────────────

def load_embedding_matrix(model_path: str):
    """从 GGUF 读取完整 token embedding 矩阵，返回 (n_embd, numpy_array[n_vocab, n_embd])。"""
    try:
        from gguf import GGUFReader
    except ImportError:
        print("[ERROR] 需要安装 gguf 库: pip install gguf")
        sys.exit(1)

    import numpy as np

    reader = GGUFReader(model_path)

    # 查找 token_embd.weight tensor
    token_embd_tensor = None
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            token_embd_tensor = t
            break

    if token_embd_tensor is None:
        print("[ERROR] 未找到 token_embd.weight tensor")
        available = [t.name for t in reader.tensors[:10]]
        print(f"[INFO] 可用 tensor (前10): {available}")
        sys.exit(1)

    # token_embd.weight 在 GGUF 中的 shape 是 [n_embd, n_vocab]
    n_embd, n_vocab = int(token_embd_tensor.shape[0]), int(token_embd_tensor.shape[1])
    print(f"[INFO] token_embd.weight: {n_embd} x {n_vocab} (n_embd x n_vocab)")

    embd_data = np.array(token_embd_tensor.data, dtype=np.float32)
    # reshape 为 [n_vocab, n_embd]，以便用 token_id 作为索引
    embd_data = embd_data.reshape(n_vocab, n_embd)
    return n_embd, embd_data


# ─── 二进制输出 ─────────────────────────────────────────────────

def write_binary(
    output_path: str,
    n_embd: int,
    char_to_tokens: dict[str, list[int]],
    embd_matrix,  # numpy [n_vocab, n_embd]
):
    """写入 HanziMap::from_embedded_bytes() 兼容的二进制文件。"""
    import numpy as np

    n_vocab = embd_matrix.shape[0]
    entries: list[tuple[str, int, list[float]]] = []

    for ch, token_ids in char_to_tokens.items():
        # 取所有 token embeddings 的平均值
        vecs = []
        for tid in token_ids:
            if tid < n_vocab:
                vecs.append(embd_matrix[tid])
        if vecs:
            avg_emb = np.mean(vecs, axis=0)
            entries.append((ch, token_ids[0], avg_emb.tolist()))

    n_chars = len(entries)
    print(f"[INFO] 写入 {n_chars} 个汉字的嵌入向量到 {output_path}")

    with open(output_path, "wb") as f:
        f.write(struct.pack("<II", n_chars, n_embd))
        for ch, token_id, emb in entries:
            char_utf32 = ord(ch)
            f.write(struct.pack("<II", char_utf32, token_id))
            f.write(struct.pack(f"<{n_embd}f", *emb))

    file_size = os.path.getsize(output_path)
    expected = 8 + n_chars * (8 + n_embd * 4)
    print(f"[INFO] 文件大小: {file_size} bytes (预期 {expected})")
    if file_size != expected:
        print(f"[WARN] 文件大小不匹配! got={file_size}, expected={expected}")

    # 显示前几个用于验证
    print(f"[INFO] 前 5 个字符:")
    for ch, token_id, emb in entries[:5]:
        print(f"  '{ch}' (U+{ord(ch):04X}) token_ids={char_to_tokens[ch]} emb[:4]={emb[:4]}")

    print(f"[INFO] 完成!")


# ─── 主流程 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="导出汉字嵌入向量 (v2 - 使用 llama-tokenize)")
    parser.add_argument("--model", required=True, help="GGUF 模型路径")
    parser.add_argument("--output", required=True, help="输出二进制文件路径")
    parser.add_argument("--chars-file", default=None, help="hanzi_table_data.rs 路径")
    parser.add_argument(
        "--tokenize-tool",
        default=None,
        help="llama-tokenize 工具路径 (自动检测)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量分词模式 (一次性传入所有字符，适用于 1:1 映射的 tokenizer)",
    )
    args = parser.parse_args()

    # ── 自动检测路径 ──
    script_dir = Path(__file__).parent

    if args.chars_file is None:
        candidates = [
            script_dir / ".." / "crates" / "xiang-core" / "src" / "hanzi_table_data.rs",
            Path("crates/xiang-core/src/hanzi_table_data.rs"),
        ]
        for c in candidates:
            if c.exists():
                args.chars_file = str(c.resolve())
                print(f"[INFO] 自动检测字符源: {args.chars_file}")
                break
        else:
            print("[ERROR] 未找到 hanzi_table_data.rs，请用 --chars-file 指定")
            sys.exit(1)

    if args.tokenize_tool is None:
        candidates = [
            script_dir / ".." / "llama.cpp" / "build" / "bin" / "Release" / "llama-tokenize.exe",
            Path("llama.cpp/build/bin/Release/llama-tokenize.exe"),
            Path("llama.cpp/build/bin/llama-tokenize"),
        ]
        for c in candidates:
            if c.exists():
                args.tokenize_tool = str(c.resolve())
                print(f"[INFO] 自动检测 tokenize 工具: {args.tokenize_tool}")
                break
        else:
            print("[ERROR] 未找到 llama-tokenize，请用 --tokenize-tool 指定")
            sys.exit(1)

    # ── 提取字符列表 ──
    chars = extract_chars_from_rust_source(args.chars_file)
    if not chars:
        print("[ERROR] 未提取到任何汉字")
        sys.exit(1)

    print(f"[INFO] 共 {len(chars)} 个汉字待导出")

    # ── 分词 ──
    if args.batch:
        print("[INFO] 批量分词模式 (所有字符一次性传入)")
        char_to_tokens = tokenize_batch(args.tokenize_tool, args.model, chars)
    else:
        print("[INFO] 逐个分词模式 (每个字符独立分词)")
        char_to_tokens = tokenize_chars(args.tokenize_tool, args.model, chars)

    matched = len(char_to_tokens)
    if matched == 0:
        print("[ERROR] 未能分词任何汉字!")
        sys.exit(1)

    multi_token = sum(1 for ids in char_to_tokens.values() if len(ids) > 1)
    print(f"[INFO] 分词结果: {matched}/{len(chars)} 成功, 其中 {multi_token} 个多 token 字符")

    # ── 读取嵌入矩阵 ──
    n_embd, embd_matrix = load_embedding_matrix(args.model)

    # ── 输出 ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_binary(args.output, n_embd, char_to_tokens, embd_matrix)

    print(f"\n[SUMMARY]")
    print(f"  模型: {args.model}")
    print(f"  嵌入维度: {n_embd}")
    print(f"  字表汉字数: {len(chars)}")
    print(f"  成功分词: {matched}")
    print(f"  输出: {args.output}")


def tokenize_batch(tokenize_tool: str, model_path: str, chars: list[str]) -> dict[str, list[int]]:
    """
    批量分词：将所有字符合并为一个字符串传入。
    仅适用于 1:1 映射的 tokenizer（即每个汉字 = 一个 token）。
    """
    combined = "".join(chars)
    result = subprocess.run(
        [tokenize_tool, "--model", model_path, "--no-bos", "--prompt", combined],
        capture_output=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
    )
    output = result.stderr + result.stdout
    token_ids = []
    for line in output.split("\n"):
        stripped = line.strip()
        if "->" in stripped:
            parts = stripped.split()
            if parts and parts[0].isdigit():
                token_ids.append(int(parts[0]))

    print(f"[INFO] 批量分词: {len(chars)} 个字符 → {len(token_ids)} 个 token")

    char_to_tokens: dict[str, list[int]] = {}
    if len(token_ids) == len(chars):
        # 1:1 映射
        for ch, tid in zip(chars, token_ids):
            char_to_tokens[ch] = [tid]
        print(f"[INFO] 完美 1:1 映射")
    else:
        # 需要逐个处理
        print("[WARN] 非 1:1 映射，改为逐个分词")
        return tokenize_chars(tokenize_tool, model_path, chars)

    return char_to_tokens


if __name__ == "__main__":
    main()
