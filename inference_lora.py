import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ========================
# コマンドライン引数設定
# ========================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_model",
    type=str,
    default=None,
    help="ベースモデルのパス。指定なしならカレントディレクトリの qwen_local/Qwen2.5-0.5B-Instruct を使用"
)
parser.add_argument(
    "--adapter_dir",
    type=str,
    default="./outputs_lora",
    help="LoRA adapter のディレクトリ"
)
parser.add_argument(
    "--prompt",
    type=str,
    required=True,
    help="生成する文章のプロンプト"
)
args = parser.parse_args()

# ========================
# モデルパスの設定
# ========================
BASE_ID = args.base_model if args.base_model is not None else os.path.join(os.getcwd(), "qwen_local", "Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = args.adapter_dir
PROMPT = args.prompt

# ========================
# トークナイザ & モデル読み込み
# ========================
tokenizer = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)

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

