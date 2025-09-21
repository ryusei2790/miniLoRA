# train_lora_lambda.py
import os, json
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===== 設定（Lambda Cloud向け）=====
# 環境変数で上書き可: export MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ID   = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")  # HFから自動DL or ローカルディレクトリ
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs_lora")
DATA_FILE  = os.environ.get("DATA_FILE", "/home/ubuntu/data/train.jsonl")  # ここにアップしておく
MAX_LEN    = int(os.environ.get("MAX_LEN", "1024"))

use_cuda = torch.cuda.is_available()
use_mps  = False  # LambdaはMPS不要

# 速度/安定化（A100想定）
if use_cuda:
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ===== Tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===== Dataset（chat→textに整形）=====
# data/train.jsonl は {"messages":[{"role":"system|user|assistant","content":"..."} , ...]} の行を想定
dataset = load_dataset("json", data_files=DATA_FILE)["train"]

def to_text(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    tokens = tokenizer(text, truncation=True, max_length=MAX_LEN)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_ds = dataset.map(to_text, remove_columns=dataset.column_names)

# ===== モデルロード（CUDAはbf16）=====
kwargs = {}
if use_cuda:
    kwargs.update(dict(torch_dtype=torch.bfloat16, device_map="auto"))
else:
    kwargs.update(dict(torch_dtype=torch.float32))

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)

# LoRA前の前処理（非量子化でもOK）
base_model = prepare_model_for_kbit_training(base_model)

# 勾配チェックポイント関連
base_model.gradient_checkpointing_enable()
base_model.enable_input_require_grads()
base_model.config.use_cache = False

# ===== LoRA設定（FFN + Attentionを推奨）=====
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",   # Attention
        "gate_proj","up_proj","down_proj"      # MLP
    ]
)
model = get_peft_model(base_model, lora)

# 学習対象の確認（任意）
model.print_trainable_parameters()

# ===== 学習ハイパラ（A100/bf16向け）=====
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=80,
    logging_steps=10,
    save_steps=80,
    bf16=use_cuda,                # CUDAならbf16
    fp16=False,                   # bf16優先
    gradient_checkpointing=True,
    dataloader_pin_memory=True,   # CUDAならTrueでOK
    dataloader_num_workers=2,
    optim="adamw_torch",
    report_to="none",
    save_safetensors=True
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collator,
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Done. LoRA adapter saved to:", OUTPUT_DIR)
