import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_ID = "/Users/ryusei/project/mr_seino/qwen_local/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "./outputs_lora"


tokenizer = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(BASE_ID, device_map="auto", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)

def chat(messages, max_new_tokens=128):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.7)
    return tokenizer.decode(out[0], skip_special_tokens=True)

messages = [
    {"role":"system","content":"あなたは丁寧で簡潔に答える日本語アシスタントです。"},
    {"role":"user","content":"敬語に書き換えて: 明日いけますか？"}
]
print(chat(messages))
