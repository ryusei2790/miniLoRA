# merge_and_save_qwen.py
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ========================
# コマンドライン引数
# ========================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_model",
    type=str,
    default=None,
    help="学習前のQwenローカルパス。指定なければカレントディレクトリの qwen_local/Qwen2.5-0.5B-Instruct を使用"
)
parser.add_argument(
    "--adapter_dir",
    type=str,
    default="./outputs_lora",
    help="学習済みLoRAディレクトリ"
)
parser.add_argument(
    "--merged_dir",
    type=str,
    default="./merged_qwen",
    help="マージ後のモデル保存先ディレクトリ"
)
args = parser.parse_args()

# ========================
# パス設定
# ========================
BASE_ID = args.base_model if args.base_model is not None else os.path.join(os.getcwd(), "qwen_local", "Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = args.adapter_dir
MERGED_DIR = args.merged_dir

# ========================
# トークナイザとベースモデルの読み込み
# ========================
tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True, trust_remote_code=True)

base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=torch.float16,        # CPUなら torch.float32 でもOK
    device_map="auto",                # GPU余裕なければ {"": "cpu"}
    trust_remote_code=True
)

# ========================
# LoRA を差し込んでマージ
# ========================
model = PeftModel.from_pretrained(base, ADAPTER_DIR)

# LoRA を焼き込み（統合）
merged = model.merge_and_unload()  # progressbar=True も可

# 保存（単体モデルとして使える）
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tok.save_pretrained(MERGED_DIR)

print("Merged to:", MERGED_DIR)
