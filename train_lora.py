import os, json
from datasets import load_dataset
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===== 設定 =====
MODEL_ID = "/Users/ryusei/project/mr_seino/qwen_local/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./outputs_lora"
DATA_FILE = "data/train.jsonl"
MAX_LEN = 1024

use_cuda = torch.cuda.is_available()
use_mps  = (torch.backends.mps.is_available() and not use_cuda)

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

# ===== モデルロード（MPS/CPUは bfloat16、CUDAならお好みで量子化）=====
kwargs = {}
if use_cuda:
    # CUDA環境なら必要に応じて bitsandbytes を使ってOK（ここではシンプルに非量子化）
    kwargs.update(dict(torch_dtype=torch.bfloat16, device_map="auto"))
elif use_mps:
    kwargs.update(dict(torch_dtype=torch.bfloat16))  # ★ MPSはbf16推奨
else:
    kwargs.update(dict(torch_dtype=torch.float32))   # ★ CPUはfp32安定

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)

# （量子化しない場合でも）学習前処理は便利なので呼んでOK
base_model = prepare_model_for_kbit_training(base_model)

# ===== 勾配チェックポイントお作法 =====
base_model.gradient_checkpointing_enable()
base_model.enable_input_require_grads()     # ★重要：これがないと勾配がNoneになりやすい
base_model.config.use_cache = False         # ★重要：CPと排他

# ===== LoRA設定（FFNのみ）=====
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none",
    target_modules=["gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(base_model, lora)

# ===== 本当に学習対象があるか確認（デバッグ用に残すと安心）=====
model.print_trainable_parameters()
for n, p in model.named_parameters():
    if p.requires_grad and "lora_" in n:
        print("[TRAINABLE]", n, tuple(p.shape))

# ===== 学習ハイパラ =====
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=80,
    logging_steps=10,
    save_steps=80,
    bf16=(use_cuda or use_mps),  # ★ CUDA/MPSならbf16を使う
    fp16=False,                  # ★ CPU/MPSではFalseが安全（CUDAでもbf16優先）
    gradient_checkpointing=True,
    dataloader_pin_memory=False, # ★ MPSでの警告回避
    optim="adamw_torch",         # CUDAで8bit optimを使うなら paged_adamw_8bit に
    report_to="none"
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collator,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Done. LoRA adapter saved to:", OUTPUT_DIR)
