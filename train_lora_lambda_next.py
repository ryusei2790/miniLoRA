# train_lora_lambda.py
import os, json
import argparse
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===== argparseでコマンドライン引数を受け取る =====
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, required=True, help="学習データ train.jsonl のパス")
parser.add_argument("--output_dir", type=str, default="./outputs_lora", help="出力先ディレクトリ")
parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HF Hub またはローカルモデルパス")
parser.add_argument("--max_len", type=int, default=1024, help="トークン最大長")
args = parser.parse_args()

DATA_FILE  = args.train_file
OUTPUT_DIR = args.output_dir
MODEL_ID   = args.model_name_or_path
MAX_LEN    = args.max_len

# ===== デバイス設定 =====
use_cuda = torch.cuda.is_available()
use_mps  = False  # Lambdaは不要

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
dataset = load_dataset("json", data_files=DATA_FILE)["train"]

def to_text(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    tokens = tokenizer(text, truncation=True, max_length=MAX_LEN)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_ds = dataset.map(to_text, remove_columns=dataset.column_names)

# ===== モデルロード =====
kwargs = {}
if use_cuda:
    kwargs.update(dict(torch_dtype=torch.bfloat16, device_map="auto"))
else:
    kwargs.update(dict(torch_dtype=torch.float32))

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
base_model = prepare_model_for_kbit_training(base_model)

base_model.gradient_checkpointing_enable()
base_model.enable_input_require_grads()
base_model.config.use_cache = False

# ===== LoRA設定 =====
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ]
)
model = get_peft_model(base_model, lora)
model.print_trainable_parameters()

# ===== 学習ハイパラ =====
args_trainer = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=80,
    logging_steps=10,
    save_steps=80,
    bf16=use_cuda,
    fp16=False,
    gradient_checkpointing=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=2,
    optim="adamw_torch",
    report_to="none",
    save_safetensors=True
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args_trainer,
    train_dataset=train_ds,
    data_collator=collator,
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Done. LoRA adapter saved to:", OUTPUT_DIR)
