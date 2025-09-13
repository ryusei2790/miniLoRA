# merge_and_save_qwen.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === ここを学習前のQwenローカルパスに ===
BASE_ID = "/Users/ryusei/project/mr_seino/qwen_local/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "./outputs_lora"     # 学習で出たLoRAのディレクトリ（adapter_config.json等がある場所）
MERGED_DIR  = "./merged_qwen"      # 保存先

# 量子化なしでロード（マージは非量子化が前提）
tok  = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True, trust_remote_code=True)

# GPU余裕→ device_map="auto" / 余裕なし→ CPUで
base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=torch.float16,          # CPUしかないなら torch.float32 でもOK
    device_map="auto",                  # きつければ {"": "cpu"}
    trust_remote_code=True
)

# LoRA を差し込み
model = PeftModel.from_pretrained(base, ADAPTER_DIR)

# LoRA を焼き込み（統合）
merged = model.merge_and_unload()       # progressbar=True も可

# 保存（これで単体モデルとして使える）
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tok.save_pretrained(MERGED_DIR)

print("Merged to:", MERGED_DIR)
