## How to Use `build_jsonl.py`

長いテキストファイルを JSONL (1行1サンプル) に変換し、学習用データを作成できます。  
以下の例を `howToUseBuild_jsonl.py` にまとめておくと便利です。

### 1. 要約タスク（assistantは空、後で学習で生成させる）
```bash
python build_jsonl.py \
  --input long.txt \
  --output data/train.jsonl \
  --model-id "/Users/ryusei/project/mr_seino/qwen_local/Qwen2.5-0.5B-Instruct" \
  --task summarize \
  --chunk-tokens 400 \
  --fill-assistant ""

python build_jsonl.py \
  --input casual.txt \
  --output data/train.jsonl \
  --model-id "/Users/ryusei/project/mr_seino/qwen_local/Qwen2.5-0.5B-Instruct" \
  --task polite \
  --chunk-tokens 350 \
  --fill-assistant placeholder

python build_jsonl.py \
  --input corpus.txt \
  --output data/train.jsonl \
  --model-id "/Users/ryusei/project/mr_seino/qwen_local/Qwen2.5-0.5B-Instruct" \
  --task identity \
  --chunk-tokens 480 \
  --fill-assistant same
