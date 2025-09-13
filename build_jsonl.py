#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
長いテキスト(.txt / .md など)を、学習用の JSONL に分解出力します。
- 例と同じ {"messages":[{role, content}, ...]} 形式で1行1サンプル
- モデルのトークナイザでおおよそMAXトークン以下に安全分割
- 用途別テンプレート (summarize / polite / identity) を用意
"""

import argparse
import json
import os
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
SYSTEM_DEFAULT = "あなたは丁寧で簡潔に答える日本語アシスタントです。"

def make_messages(task: str, chunk_text: str, system_prompt: str, fill_assistant: str) -> Dict:
    """
    task: summarize / polite / identity / custom
    fill_assistant:
      - ""         : 空(教師なしの雛形づくり。後で埋める)
      - "same"     : 入力と同じ内容をassistantに入れる（写経/自己復元学習など）
      - "placeholder" : 「[[ANSWER]]」というプレースホルダ文字列を入れる
    """
    if task == "summarize":
        user_content = f"要約: 次の文章を短く簡潔にまとめて。\n\n{chunk_text}"
    elif task == "polite":
        user_content = f"敬語に書き換えて: 次の文章を丁寧語・敬体に自然に整えて。\n\n{chunk_text}"
    elif task == "identity":
        user_content = f"整形: 次の文章をそのまま出力してください（体裁の崩れのみ最小限で整形）。\n\n{chunk_text}"
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
# メイン
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="長文を学習用JSONLに分解するツール")
    ap.add_argument("--input", required=True, help="入力テキストファイル(.txt/.mdなど)")
    ap.add_argument("--output", default="data/train.jsonl", help="出力JSONLパス (既定: data/train.jsonl)")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct",
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
    args = ap.parse_args()

    # 読み込み
    with open(args.input, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        print("⚠️ 入力ファイルが空です。処理を終了します。", file=sys.stderr)
        sys.exit(1)

    # トークナイザ
    tokenizer = AutoTokenizer.from_pretrained(args.model-id if hasattr(args, "model-id") else args.model_id, use_fast=True)

    # 分割
    chunks = chunk_by_tokens(
        text=raw,
        tokenizer=tokenizer,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens
    )

    # JSONLに書き出し
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    num_written = 0
    with open(args.output, "w", encoding="utf-8") as w:
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
