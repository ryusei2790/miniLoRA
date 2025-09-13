# LocalLLMLoRA

ローカル環境でLLMのLoRA（Low-Rank Adaptation）ファインチューニングを行うためのプロジェクトです。長いテキストファイルから学習用データを作成し、LoRAアダプターを学習して推論を行うことができます。

## 目次

- [インストール](#インストール)
- [プロジェクト構成](#プロジェクト構成)
- [使用方法](#使用方法)
  - [1. 学習データの準備](#1-学習データの準備)
  - [2. LoRAファインチューニング](#2-loraファインチューニング)
  - [3. 推論](#3-推論)
- [モデルの追加方法](#モデルの追加方法)
- [設定のカスタマイズ](#設定のカスタマイズ)
- [トラブルシューティング](#トラブルシューティング)

## インストール

### 必要な環境
- Python 3.8以上
- CUDA対応GPU（推奨）

### 依存関係のインストール

```bash
# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate     # Windows

# 必要なパッケージのインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install peft
pip install accelerate
pip install datasets
```

## プロジェクト構成

```
LocalLLMLoRA/
├── build_jsonl.py          # テキストファイルをJSONL形式に変換
├── inference_lora.py       # LoRAモデルでの推論
├── example_data.jsonl      # サンプルデータ
├── data/                   # 学習データディレクトリ
├── outputs_lora/           # LoRAアダプターの出力ディレクトリ
├── merged_qwen/            # マージされたモデルディレクトリ
└── venv/                   # 仮想環境
```

## 使用方法

### 1. 学習データの準備

`build_jsonl.py`を使用して、長いテキストファイルを学習用のJSONL形式に変換します。

#### 基本的な使用方法

```bash
python build_jsonl.py \
  --input your_text_file.txt \
  --output data/train.jsonl \
  --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
  --task summarize \
  --chunk-tokens 400
```
```bash
python build_jsonl.py \
  --input data/long.txt \
  --output data/train.jsonl \
  --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
  --task summarize \
  --chunk-tokens 400
```
#### パラメータの説明

- `--input`: 入力テキストファイル（.txt、.mdなど）
- `--output`: 出力JSONLファイルのパス
- `--model-id`: トークナイザーに使用するモデルID（Hugging Face Hubまたはローカルパス）
- `--task`: タスクの種類（`summarize`, `polite`, `identity`, `custom`）
- `--chunk-tokens`: 1チャンクあたりの最大トークン数
- `--system`: システムプロンプト（デフォルト: "あなたは丁寧で簡潔に答える日本語アシスタントです。"）
- `--fill-assistant`: assistant部分の埋め方（`""`（空）, `"same"`（同文）, `"placeholder"`）

#### タスクの種類

1. **要約タスク**（`summarize`）
```bash
python build_jsonl.py \
  --input long.txt \
  --output data/train.jsonl \
  --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
  --task summarize \
  --chunk-tokens 400 \
  --fill-assistant ""
```

2. **敬語変換タスク**（`polite`）
```bash
python build_jsonl.py \
  --input casual.txt \
  --output data/train.jsonl \
  --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
  --task polite \
  --chunk-tokens 350 \
  --fill-assistant placeholder
```

3. **同一性タスク**（`identity`）
```bash
python build_jsonl.py \
  --input corpus.txt \
  --output data/train.jsonl \
  --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
  --task identity \
  --chunk-tokens 480 \
  --fill-assistant same
```

### 2. LoRAファインチューニング

学習データが準備できたら、LoRAファインチューニングを実行します。

```bash
# accelerateを使用した学習（推奨）
accelerate launch --config_file accelerate_config.yaml train_lora.py \
  --model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct" \
  --train_file "data/train.jsonl" \
  --output_dir "outputs_lora" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --lora_rank 16 \
  --lora_alpha 32
```

### 3. 推論

学習済みのLoRAアダプターを使用して推論を行います。

```python
# inference_lora.pyを実行
python inference_lora.py
```

または、Pythonスクリプト内で直接使用：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ベースモデルとLoRAアダプターの読み込み
BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "./outputs_lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_ID, 
    device_map="auto", 
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)

# 推論実行
messages = [
    {"role": "system", "content": "あなたは丁寧で簡潔に答える日本語アシスタントです。"},
    {"role": "user", "content": "敬語に書き換えて: 明日いけますか？"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128, 
        do_sample=True, 
        top_p=0.9, 
        temperature=0.7
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## モデルの追加方法

### 新しいベースモデルの使用

1. **Hugging Face Hubからモデルをダウンロード**
```bash
# モデルをローカルにダウンロード
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
```

2. **inference_lora.pyの設定を変更**
```python
# ベースモデルのパスを変更
BASE_ID = "/path/to/your/model"  # ローカルパスまたはHugging Face ID
```

3. **build_jsonl.pyでモデルIDを指定**
```bash
python build_jsonl.py \
  --input your_file.txt \
  --output data/train.jsonl \
  --model-id "/path/to/your/model" \
  --task summarize
```

### 対応モデル

このプロジェクトは以下のモデルで動作確認済みです：

- Qwen2.5-0.5B-Instruct
- Qwen2.5-1.5B-Instruct
- Qwen2.5-3B-Instruct
- Llama-2-7B-Chat
- Mistral-7B-Instruct

その他のモデルも、ChatML形式または類似のチャットテンプレートに対応していれば使用可能です。

## 設定のカスタマイズ

### LoRAパラメータの調整

学習時に以下のパラメータを調整できます：

- `--lora_rank`: LoRAのランク（デフォルト: 16）
- `--lora_alpha`: LoRAのスケーリングパラメータ（デフォルト: 32）
- `--lora_dropout`: LoRAのドロップアウト率（デフォルト: 0.1）
- `--target_modules`: 対象とするモジュール（デフォルト: ["q_proj", "v_proj"]）

### 学習パラメータの調整

- `--learning_rate`: 学習率（推奨: 1e-4 ～ 5e-4）
- `--num_train_epochs`: エポック数（推奨: 1-5）
- `--per_device_train_batch_size`: バッチサイズ
- `--gradient_accumulation_steps`: 勾配累積ステップ数

## トラブルシューティング

### よくある問題と解決方法

1. **CUDA out of memory エラー**
   - バッチサイズを小さくする
   - 勾配累積ステップ数を増やす
   - より小さなモデルを使用する

2. **トークナイザーエラー**
   - モデルIDが正しいか確認
   - ローカルパスの場合は、モデルファイルが存在するか確認

3. **学習が進まない**
   - 学習率を調整する
   - データの品質を確認する
   - エポック数を増やす

### ログの確認

学習中のログを確認して、損失の変化や学習の進捗を監視してください。

```bash
# 学習ログの確認
tail -f outputs_lora/training_log.txt
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

バグ報告や機能追加の提案は、GitHubのIssuesまたはPull Requestsでお願いします。