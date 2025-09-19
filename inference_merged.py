import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./merged_qwen"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)

def chat(messages, max_new_tokens=512):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.7)
    return tokenizer.decode(out[0], skip_special_tokens=True)

messages = [
    {"role":"system","content":"あなたは感動的な文章を書く小説家です。"},
    {"role":"user","content":"感動的で人情に訴えかけてくるような新しい小説を作成して"}
]
print(chat(messages))
