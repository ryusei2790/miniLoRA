# LocalLLMLoRA

ローカル環境でLLMのLoRA（Low-Rank Adaptation）ファインチューニングを行うためのプロジェクトです。長いテキストファイルから学習用データを作成し、LoRAアダプターを学習して推論を行うことができます。

## 目次

* [インストール](#インストール)
* [SSH接続してLambdaLabsの実行スタート](#ssh接続してlambdalabsの実行スタート)
* [クラウドGPU環境での実行](#クラウドgpu環境での実行)
* [プロジェクト構成](#プロジェクト構成)
* [使用方法](#使用方法)

  * [1. 学習データの準備](#1-学習データの準備)
  * [2. LoRAファインチューニング](#2-loraファインチューニング)
  * [3. 推論](#3-推論)
* [Lambda Labs 実行メモ（HFキャッシュ先とデータパス）](#lambda-labs-実行メモhfキャッシュ先とデータパス)
* [LoRA vs マージ方式の比較](#lora-vs-マージ方式の比較)
* [モデルの追加方法](#モデルの追加方法)
* [設定のカスタマイズ](#設定のカスタマイズ)
* [トラブルシューティング](#トラブルシューティング)

## SSH接続してLambdaLabsの実行スタート

```bash 
ssh -i ~/.ssh/id_ed25519 ubuntu@<インスタンスのIPアドレス>

```

## インストール

### 必要な環境

* Python 3.8以上（3.10推奨）
* CUDA対応GPU（推奨）

### 依存関係のインストール

```bash
# Gitとvenvをインストール（Ubuntu）
sudo apt-get update -y && sudo apt-get install -y git python3.10-venv

# リポジトリをクローン（あなたのユーザー名に置き換え）
git clone https://github.com/<YOUR_USERNAME>/
cd miniLoRA   # or miniLoRA（あなたの環境に合わせて）

# 仮想環境の作成（推奨）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# または（Windows）
# venv\\Scripts\\activate

# データセット用ディレクトリ
mkdir -p data
cd ~/miniLoRA/data

# サンプルの空ファイル
touch long.txt   # ← 以前の 'long.txst' はタイポ

# テキスト編集（必要なら）
nano long.txt
# 保存は Ctrl+O → Enter → Ctrl+X

# 必要なパッケージのインストール（CUDA 11.8 の例）
pip install -U pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft accelerate datasets
```

## クラウドGPU環境での実行

このプロジェクトはクラウドGPU環境（AWS、GCP、Azure、RunPod、Lambda Labs等）で実行できます。

### 推奨スペック

#### 学習（LoRA、Qwen2.5-0.5B級）

* **GPU**: 24GB VRAM以上（A5000/RTX 6000/A10/A100等）
* **CPU**: 4コア以上
* **RAM**: 32GB以上
* **ストレージ**: 50GB以上

#### 推論のみ

* **FP16**: 16-24GB VRAM
* **4bit量子化**: 8-12GB VRAM（`bitsandbytes`使用時）

### セットアップ手順（Ubuntu系インスタンス）

```bash
# 1. ドライバとGPU確認
nvidia-smi | cat

# 2. Python & venv
sudo apt-get update -y && sudo apt-get install -y python3.10-venv git
python3 -m venv .venv
source .venv/bin/activate

# 3. pipの高速化
pip install -U pip setuptools wheel

# 4. 依存関係インストール
pip install -r requirements.txt  # ある場合
# or: pip install transformers peft accelerate datasets

# 5. （学習時のみ）accelerate初期化
accelerate config default
# または非対話式:
accelerate config --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
  --mixed_precision fp16 --num_processes 1 --num_machines 1 \
  --use_cpu false --dynamo_backend no
```

### 環境変数設定（推奨）

```bash
# OOM回避
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HFキャッシュはホーム配下に固定（/workspace を参照しないように）
mkdir -p /home/ubuntu/.cache/huggingface/hub
export HF_HOME=/home/ubuntu/.cache/huggingface
export HF_HUB_CACHE=/home/ubuntu/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/ubuntu/.cache/huggingface/hub
export XDG_CACHE_HOME=/home/ubuntu/.cache
```

### Hugging Face認証（推奨）

```bash
# レート制限回避・プライベートモデル取得用
huggingface-cli login --token <YOUR_HF_TOKEN> --add-to-git-credential
```

### クラウド環境での実行例

```bash
# LoRA学習
python train_lora_lambda.py \
  --train_file data/train.jsonl \
  --output_dir outputs_lora \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --bf16 True

# LoRA推論
python inference_lora.py \
  --adapter_dir outputs_lora \
  --prompt "感動的で人情に訴えかけてくるような新しい小説を作成して"

# マージ済みモデル推論
python inference_merged.py \
  --model_dir merged_qwen \
  --prompt "感動的で人情に訴えかけてくるような新しい小説を作成して"
```

## プロジェクト構成

```
LocalLLMLoRA/
├── build_jsonl.py          # テキストファイルをJSONL形式に変換
├── train_lora.py           # LoRA学習（accelerate使用版など）
├── train_lora_lambda.py    # Lambda Labs 向け簡易スクリプト
├── inference_lora.py       # LoRAモデルでの推論
├── merge_and_save.py       # LoRA→ベースにマージ
├── inference_merged.py     # マージ済みモデルでの推論
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
python3 build_jsonl.py \
  --input data/long.txt \
  --output data/train.jsonl \
  --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
  --task summarize \
  --chunk-tokens 400
```

#### パラメータの説明

* `--input`: 入力テキストファイル（.txt、.mdなど）
* `--output`: 出力JSONLファイルのパス
* `--model-id`: トークナイザーに使用するモデルID（Hugging Face Hubまたはローカルパス）
* `--task`: タスクの種類（`summarize`, `polite`, `identity`, `custom`）
* `--chunk-tokens`: 1チャンクあたりの最大トークン数
* `--system`: システムプロンプト（デフォルト: "あなたは丁寧で簡潔に答える日本語アシスタントです。"）
* `--fill-assistant`: assistant部分の埋め方（`""`（空）, `"same"`（同文）, `"placeholder"`）

#### タスクの種類

1. **チャンク化タスク**（`summarize`）

```bash
python3 build_jsonl.py \
  --input data/long.txt \
  --output data/train.jsonl \
  --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
  --task summarize \
  --chunk-tokens 400 \
  --fill-assistant ""
```

2. **変換タスク**（`polite`）

```bash
python3 build_jsonl.py \
  --input data/casual.txt \
  --output data/train.jsonl \
  --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
  --task polite \
  --chunk-tokens 350 \
  --fill-assistant placeholder
```

3. **同一性タスク**（`identity`）

```bash
python3 build_jsonl.py \
  --input data/corpus.txt \
  --output data/train.jsonl \
  --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
  --task identity \
  --chunk-tokens 480 \
  --fill-assistant same
```

> 区切り文字で文章を分けたい場合（例：`---` の行で分割）

```bash
python3 build_jsonl.py \
  --input data/long.txt \
  --output data/train.jsonl \
  --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
  --task identity \
  --chunk-tokens 480 \
  --fill-assistant same \
  --delimiter '^---$' \
  --delimiter-regex
```

### 2. LoRAファインチューニング

学習データが準備できたら、LoRAファインチューニングを実行します。

```bash
# 事前にベース精度を確認しておくと比較に便利
python3 inference_lora.py --adapter_dir outputs_lora --prompt "こんにちは"

# accelerateを使用した学習（例）
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

#### 3-1. LoRAアダプターでの推論

```bash
python3 inference_lora.py --adapter_dir outputs_lora --prompt "敬語に書き換えて: 明日いけますか？"
```

#### 3-2. マージされたモデルでの推論

```bash
# LoRAアダプターをベースモデルに統合
python3 merge_and_save.py

# マージ済みモデルで推論
python3 inference_merged.py --model_dir merged_qwen --prompt "敬語に書き換えて: 明日いけますか？"
```

---

## Lambda Labs 実行メモ（HFキャッシュ先とデータパス）

**背景**: 一部の Lambda Labs イメージでは既定で `HF_HOME=/workspace/.cache/huggingface` になっており、`/workspace` が存在しない/権限が無いと **`PermissionError: '/workspace'`** が発生します。以下は実際に解決した手順のまとめです。

### 1) 現状確認

```bash
printenv | egrep 'HF|HUGGINGFACE|TRANSFORMERS|XDG'
ls -ld /workspace  # 無い/アクセス不可なら次へ
```

### 2) キャッシュ先をホーム配下に切替（このシェルだけで有効）

```bash
mkdir -p /home/ubuntu/.cache/huggingface/hub
chmod -R 700 /home/ubuntu/.cache

export HF_HOME=/home/ubuntu/.cache/huggingface
export HF_HUB_CACHE=/home/ubuntu/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/ubuntu/.cache/huggingface/hub
export XDG_CACHE_HOME=/home/ubuntu/.cache

# 反映確認
printenv HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
```

> 注: 実行時に `FutureWarning: Using TRANSFORMERS_CACHE is deprecated ... Use HF_HOME instead` が出る場合があります。`HF_HOME` を設定していれば問題ありません。

### 3) ロックファイルの削除（必要に応じて）

```bash
find /home/ubuntu/.cache/huggingface -name "*.lock" -type f -delete
```

### 4) FileNotFoundError（データファイル未設置）の対処

```bash
# まず存在確認
ls -l /home/ubuntu/data/train.jsonl || ls -l ~/LocalLLMLoRA/data/train.jsonl

# 正しい場所をDATA_FILEで渡す
export DATA_FILE=/home/ubuntu/LocalLLMLoRA/data/train.jsonl  # 実在パスに合わせて
python3 train_lora_lambda.py
```

#### 最小JSONLを作って動作確認する

```bash
mkdir -p /home/ubuntu/data
cat > /home/ubuntu/data/train.jsonl <<'EOF'
{"messages":[
  {"role":"system","content":"あなたは丁寧な日本語アシスタントです。"},
  {"role":"user","content":"こんにちは"},
  {"role":"assistant","content":"こんにちは！今日は何をしますか？"}
]}
{"messages":[
  {"role":"system","content":"あなたは丁寧な日本語アシスタントです。"},
  {"role":"user","content":"自己紹介して"},
  {"role":"assistant","content":"はい、よろしくお願いします。"}
]}
EOF

export DATA_FILE=/home/ubuntu/data/train.jsonl
python3 train_lora_lambda.py
```

#### さらに堅くする（コード側で `cache_dir` を明示）

```python
# train_lora_lambda.py の抜粋
CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, cache_dir=CACHE_DIR)
dataset   = load_dataset("json", data_files=DATA_FILE, cache_dir=CACHE_DIR)["train"]
base_model= AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, **kwargs)
```

---

## LoRA vs マージ方式の比較

このプロジェクトでは2つのアプローチを提供しています：

### 1. LoRAアダプター方式（上書き・追加学習）

**動作原理**:

* 元モデルの重みは**そのまま保持**
* LoRAアダプタ（小さな追加パラメータ）を学習
* 推論時は元モデル + LoRAアダプタを動的に組み合わせ

**メリット**:

* ファイルサイズが小さい（数MB〜数十MB）
* 複数のタスクに対応可能（アダプタを切り替え）
* メモリ効率が良い
* 元モデルの能力を完全に保持

**デメリット**:

* 推論時にベースモデル + アダプタ両方必要
* 読み込み時間が長い
* デプロイが複雑

**用途**: 開発・実験段階、複数タスクの切り替え

### 2. マージ方式（完全統合）

**動作原理**:

* LoRAアダプタの重みを元モデルに**完全統合**
* 新しい単体モデルとして保存
* 推論時は統合済みモデルのみ使用

**メリット**:

* 単体モデルで動作
* 読み込みが高速
* デプロイが簡単
* 配布しやすい

**デメリット**:

* ファイルサイズが大きい（元モデルと同じサイズ）
* 特定タスクに固定
* 元モデルは変更される

**用途**: 本番環境、単一タスクの運用

### 具体的な比較

| 項目          | LoRAアダプター     | マージ方式        |
| ----------- | ------------- | ------------ |
| **ファイルサイズ** | 小さい（数MB〜数十MB） | 大きい（元モデルと同じ） |
| **推論速度**    | やや遅い（組み合わせ処理） | 高速（単体モデル）    |
| **メモリ使用量**  | 効率的           | 標準的          |
| **デプロイ**    | 複雑（2つのファイル）   | 簡単（1つのファイル）  |
| **タスク切り替え** | 可能（アダプタ交換）    | 不可（固定）       |
| **元モデル保持**  | 完全保持          | 統合される        |

### 推奨ワークフロー

1. **開発段階**: LoRA方式で実験・調整
2. **本番準備**: マージ方式で統合モデル作成
3. **配布**: マージ済みモデルを配布

## モデルの追加方法

### 新しいベースモデルの使用

1. **Hugging Face Hubからモデルをダウンロード**

```bash
# モデルをローカルにダウンロード
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
```

2. **inference\_lora.pyの設定を変更**

```python
# ベースモデルのパスを変更
BASE_ID = "/path/to/your/model"  # ローカルパスまたはHugging Face ID
```

3. **build\_jsonl.pyでモデルIDを指定**

```bash
python3 build_jsonl.py \
  --input your_file.txt \
  --output data/train.jsonl \
  --model-id "/path/to/your/model" \
  --task summarize
```

### 対応モデル

* Qwen2.5-0.5B-Instruct
* Qwen2.5-1.5B-Instruct
* Qwen2.5-3B-Instruct
* Llama-2-7B-Chat
* Mistral-7B-Instruct

その他のモデルも、ChatML形式または類似のチャットテンプレートに対応していれば使用可能です。

## 設定のカスタマイズ

### LoRAパラメータの調整

* `--lora_rank`: LoRAのランク（デフォルト: 16）
* `--lora_alpha`: LoRAのスケーリングパラメータ（デフォルト: 32）
* `--lora_dropout`: LoRAのドロップアウト率（デフォルト: 0.1）
* `--target_modules`: 対象とするモジュール（デフォルト: \["q\_proj", "v\_proj"]）

### 学習パラメータの調整

* `--learning_rate`: 学習率（推奨: 1e-4 ～ 5e-4）
* `--num_train_epochs`: エポック数（推奨: 1-5）
* `--per_device_train_batch_size`: バッチサイズ
* `--gradient_accumulation_steps`: 勾配累積ステップ数

## トラブルシューティング

### よくある問題と解決方法

1. **CUDA out of memory エラー**

   * バッチサイズを小さくする
   * 勾配累積ステップ数を増やす
   * より小さなモデルを使用する
   * `bf16=True`を使用する
   * 4bit量子化を検討する（`bitsandbytes`）

2. **トークナイザーエラー**

   * モデルIDが正しいか確認
   * ローカルパスの場合は、モデルファイルが存在するか確認
   * `trust_remote_code=True`を追加

3. **学習が進まない**

   * 学習率を調整する（1e-4 ～ 5e-4）
   * データの品質を確認する
   * エポック数を増やす
   * LoRAのrankを調整する

4. **クラウド環境での問題**

   * ドライババージョンを確認（R550+推奨）
   * PyTorchのCUDAバージョンを確認
   * ネットワーク接続を確認
   * ディスク容量を確認

### ログの確認

```bash
# 学習ログの確認（例）
tail -f outputs_lora/training_log.txt

# GPU使用状況の確認
nvidia-smi

# メモリ使用量の確認
htop
```

### デバッグ用の設定

```python
# 学習可能パラメータの確認
model.print_trainable_parameters()

# LoRAパラメータの確認
for n, p in model.named_parameters():
    if p.requires_grad and "lora_" in n:
        print("[TRAINABLE]", n, tuple(p.shape))
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 別プロジェクトでの使用

学習済みのLoRAアダプター（`outputs_lora`ディレクトリ）を他のプロジェクトで使用する方法を説明します。

### 必要なファイル

```
outputs_lora/
├── adapter_config.json       # LoRAの設定情報
├── adapter_model.safetensors # LoRAの重み
├── tokenizer.json            # トークナイザー設定
├── special_tokens_map.json   # 特殊トークン設定
├── chat_template.jinja       # チャットテンプレート
└── vocab.json                # 語彙ファイル
```

### 別プロジェクトでの使用方法

#### 1. ファイルのコピー

```bash
# 学習済みのLoRAアダプターを新しいプロジェクトにコピー
cp -r /path/to/LocalLLMLoRA/outputs_lora /path/to/new_project/
```

#### 2. 新しいプロジェクトでの実装

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ベースモデルの設定（元の学習で使用したモデルと同じものを使用）
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # またはローカルパス
ADAPTER_DIR = "./outputs_lora"  # コピーしたLoRAアダプターのパス

# トークナイザーとベースモデルの読み込み
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
)

# LoRAアダプターの適用
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# 推論関数
def generate_response(messages, max_new_tokens=128):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 使用例
messages = [
    {"role": "system", "content": "あなたは丁寧で簡潔に答える日本語アシスタントです。"},
    {"role": "user", "content": "敬語に書き換えて: 明日いけますか？"}
]

response = generate_response(messages)
print(response)
```

#### 3. 依存関係のインストール

```bash
pip install torch transformers peft accelerate
```

### 注意事項

#### ベースモデルの互換性

* **同じベースモデルを使用**: 学習時に使用したベースモデルと同じものを使用してください
* **バージョンの一致**: transformersライブラリのバージョンが大きく異なる場合は互換性の問題が発生する可能性があります

#### 設定の確認

`adapter_config.json`でベースモデルのパスを確認：

```json
{
  "base_model_name_or_path": "/Users/ryusei/project/mr_seino/qwen_local/Qwen2.5-0.5B-Instruct",
  "target_modules": ["up_proj", "gate_proj", "down_proj"],
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05
}
```

#### パスの調整

新しい環境では、`base_model_name_or_path`のパスを調整する必要がある場合があります：

```python
# ローカルパスの場合
BASE_MODEL_ID = "/path/to/your/qwen_local/Qwen2.5-0.5B-Instruct"

# Hugging Face Hubの場合
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
```

### 配布用のパッケージ化

#### 1. アダプターのみの配布

```bash
# 必要なファイルのみをアーカイブ
tar -czf lora_adapter.tar.gz \
  outputs_lora/adapter_config.json \
  outputs_lora/adapter_model.safetensors \
  outputs_lora/tokenizer.json \
  outputs_lora/special_tokens_map.json \
  outputs_lora/chat_template.jinja \
  outputs_lora/vocab.json
```

#### 2. 使用方法のドキュメント

配布時に以下の情報を含めることを推奨：

* ベースモデルの情報
* 学習データの概要
* 推奨される使用方法
* システム要件

### トラブルシューティング

#### よくある問題

1. **モデルの読み込みエラー**

   ```python
   # 解決策: ベースモデルのパスを確認
   print("Base model path:", model.config.base_model_name_or_path)
   ```

2. **トークナイザーの不一致**

   ```python
   # 解決策: ベースモデルのトークナイザーを使用
   tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
   ```

3. **デバイスの不一致**

   ```python
   # 解決策: デバイスを明示的に指定
   model = model.to("cuda" if torch.cuda.is_available() else "cpu")
   ```

## 貢献

バグ報告や機能追加の提案は、GitHubのIssuesまたはPull Requestsでお願いします。このバージョンを参照して今のやつの必要なところだけ追記して








Lambda Labs → ローカルへの LoRA ZIP ダウンロード手順
前提

Lambda Labs 上で LoRA モデルを作成済み

ZIP にまとめてある：~/miniLoRA/outputs_lora.zip

ローカル（Mac）には SSH 鍵 ~/.ssh/id_ed25519 / id_ed25519.pub がある

手順 1：ローカル端末で公開鍵を確認
# ローカル（Mac）で実行
ls ~/.ssh
cat ~/.ssh/id_ed25519.pub


公開鍵の内容が表示されれば OK

形式は ssh-ed25519 AAAAC3… user@hostname

手順 2：ローカルから Lambda Labs に公開鍵を登録
# ローカル（Mac）で実行
ssh-copy-id ubuntu@129.213.18.241


初回接続時は以下の確認が出ます → yes と入力

Are you sure you want to continue connecting (yes/no/[fingerprint])? yes


成功するとパスワードを入力した後、サーバに鍵が登録されます

これで次回以降はパスワードなしで SSH/SCP が可能

手順 3：ローカルから ZIP をダウンロード
# ローカル（Mac）で実行
scp ubuntu@129.213.18.241:~/miniLoRA/outputs_lora.zip ./outputs_lora.zip


ダウンロードが完了すると、ローカルに outputs_lora.zip が保存されます

手順 4：ローカルで ZIP を展開
# ローカル（Mac）で実行
unzip outputs_lora.zip -d ./outputs_lora


サーバ上と同じディレクトリ構造で展開されます

LoRA モデルの本体は adapter_model.safetensors