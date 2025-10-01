import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========================
# コマンドライン引数
# ========================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir",
    type=str,
    default=None,
    help="マージ済みモデルのディレクトリ。指定なければカレントディレクトリの merged_qwen を使用"
)
parser.add_argument(
    "--prompt",
    type=str,
    required=True,
    help="生成する文章のプロンプト"
)
args = parser.parse_args()

# ========================
# パス設定
# ========================
MODEL_DIR = args.model_dir if args.model_dir is not None else os.path.join(os.getcwd(), "merged_qwen")
PROMPT = args.prompt

# ========================
# トークナイザとモデルの読み込み
# ========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
)

# ========================
# チャット関数
# ========================
def chat(messages, max_new_tokens=1024):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ========================
# 実行例
# ========================
messages = [
    {"role": "system", "content": "あなたは感動的な文章を書く小説家です。"},
    {"role": "user", "content": PROMPT}
]

print(chat(messages))
