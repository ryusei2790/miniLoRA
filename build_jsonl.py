#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
長いテキスト(.txt / .md など)を、学習用の JSONL に分解出力します。
- 例と同じ {"messages":[{role, content}, ...]} 形式で1行1サンプル
- モデルのトークナイザでおおよそMAXトークン以下に安全分割
- 用途別テンプレート (summarize / polite / identity) を用意
- ★ 先に「区切り」で前分割 → 各ブロックをトークン分割（--delimiter / --delimiter-regex）
"""

import argparse
import json
import os
import re
import sys
from typing import List, Dict

try:
    from transformers import AutoTokenizer
except Exception as e:
    print("⚠️ transformers が必要です: pip install transformers", file=sys.stderr)
    raise

# -------------------------------
# テンプレート定義
# -------------------------------
SYSTEM_DEFAULT = "あなたはモダン小説家です。"

def make_messages(task: str, chunk_text: str, system_prompt: str, fill_assistant: str) -> Dict:
    """
    task: summarize / polite / identity / custom
    fill_assistant:
      - ""         : 空(教師なしの雛形づくり。後で埋める)
      - "same"     : 入力と同じ内容をassistantに入れる（写経/自己復元学習など）
      - "placeholder" : 「[[ANSWER]]」というプレースホルダ文字列を入れる
    """
    if task == "summarize":
        user_content = f"{chunk_text}"
    elif task == "polite":
        user_content = f"{chunk_text}"
    elif task == "identity":
        user_content = f"{chunk_text}"
    else:  # custom
        user_content = chunk_text

    if fill_assistant == "same":
        assistant_content = chunk_text
    elif fill_assistant == "placeholder":
        assistant_content = "[[ANSWER]]"
    else:
        assistant_content = ""

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }

# -------------------------------
# 分割ロジック（トークンベース）
# -------------------------------
def chunk_by_tokens(text: str, tokenizer, chunk_tokens: int, overlap_tokens: int = 0) -> List[str]:
    """
    トークン長でおおよそ分割。日本語の文境界に完全一致はしないが実務上は十分。
    overlap_tokens を >0 にすると隣接チャンクにトークンを重ねて文脈を持たせられる。
    """
    if chunk_tokens <= 0:
        return [text]

    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    n = len(ids)

    while start < n:
        end = min(start + chunk_tokens, n)
        piece_ids = ids[start:end]
        chunk_text = tokenizer.decode(piece_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == n:
            break
        # 次の開始位置（オーバーラップ考慮）
        start = end - max(0, overlap_tokens)

    return chunks

# -------------------------------
# ブロック前分割（区切り）
# -------------------------------
def split_blocks(raw: str, delimiter: str | None, use_regex: bool) -> List[str]:
    """
    delimiter が指定されたらその区切りで raw を前分割する。
    - use_regex=True の場合は正規表現 split（複数行境界なども可）
    - None/空 の場合は raw 全体を1ブロックで返す
    """
    if not delimiter:
        return [raw]

    if use_regex:
        # 行単位の区切り（例: ^---$）を使いたい場合は MULTILINE
        pattern = re.compile(delimiter, flags=re.MULTILINE)
        parts = re.split(pattern, raw)
    else:
        parts = raw.split(delimiter)

    # 先頭/末尾の空を除去せず、その後で空白のみのブロックを間引く
    blocks = [p.strip() for p in parts]
    return [b for b in blocks if b]  # 空は捨てる

# -------------------------------
# メイン
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="長文を学習用JSONLに分解するツール（区切り→トークン分割対応）")
    ap.add_argument("--input", required=True, help="入力テキストファイル(.txt/.mdなど)")
    ap.add_argument("--output", default="data/train.jsonl", help="出力JSONLパス (既定: data/train.jsonl)")
    ap.add_argument("--model-id", dest="model_id", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="トークナイザに使うモデルID or ローカルパス（あなたの環境のQwenローカルもOK）")
    ap.add_argument("--system", default=SYSTEM_DEFAULT, help="systemプロンプト文")
    ap.add_argument("--task", choices=["summarize", "polite", "identity", "custom"],
                    default="summarize", help="テンプレート種類")
    ap.add_argument("--chunk-tokens", type=int, default=400,
                    help="1チャンクあたりの最大トークン数（目安）")
    ap.add_argument("--overlap-tokens", type=int, default=0,
                    help="チャンク間のオーバーラップトークン数")
    ap.add_argument("--fill-assistant", choices=["", "same", "placeholder"], default="",
                    help='assistant部分の埋め方: ""(空) / "same"(同文) / "placeholder"([[ANSWER]])')
    ap.add_argument("--min-chars", type=int, default=5,
                    help="短すぎるチャンクは捨てる(文字数条件)")
    ap.add_argument("--delimiter", default="", help="前分割に使う区切り文字列。例: '---' や '||||'。未指定なら分割しない")
    ap.add_argument("--delimiter-regex", action="store_true",
                    help="--delimiter を正規表現として扱う（例: '^---$' を行単位の区切りに）")

    # ★ 新規追加
    ap.add_argument("--encoding", default="utf-8",
                    help="入力ファイルのエンコーディング (例: utf-8, shift_jis, euc-jp)")

    args = ap.parse_args()

    # 読み込み（常にUTF-8に変換）
    with open(args.input, "r", encoding=args.encoding, errors="replace") as f:
        raw = f.read()
    # 一旦UTF-8で再エンコード・再デコードして統一
    raw = raw.encode("utf-8", errors="replace").decode("utf-8").strip()

    if not raw:
        print("⚠️ 入力ファイルが空です。処理を終了します。", file=sys.stderr)
        sys.exit(1)


    # トークナイザ
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    # ★ 先にブロック前分割
    blocks = split_blocks(raw, delimiter=args.delimiter, use_regex=args.delimiter_regex)
    if not blocks:
        print("⚠️ 区切り後の有効ブロックが0件でした。delimiterの指定を見直してください。", file=sys.stderr)
        sys.exit(1)

    # JSONLに書き出し
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    num_written = 0
    with open(args.output, "w", encoding="utf-8") as w:
        for block in blocks:
            # 各ブロックをさらにトークン分割
            chunks = chunk_by_tokens(
                text=block,
                tokenizer=tokenizer,
                chunk_tokens=args.chunk_tokens,
                overlap_tokens=args.overlap_tokens
            )
            for ch in chunks:
                if len(ch.strip()) < args.min_chars:
                    continue
                obj = make_messages(
                    task=args.task,
                    chunk_text=ch.strip(),
                    system_prompt=args.system,
                    fill_assistant=args.fill_assistant
                )
                w.write(json.dumps(obj, ensure_ascii=False))
                w.write("\n")
                num_written += 1

    print(f"✅ 完了: {args.output} に {num_written} 件を書き出しました。")

if __name__ == "__main__":
    main()
