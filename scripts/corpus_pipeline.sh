#!/bin/bash
# Full 250 MB corpus pilot, end to end, writing everything into the workspace
# (the sandbox /tmp is ephemeral between sessions).
set -e
cd /workspace/reverse_dna
ROOT=/workspace/reverse_dna/corpus_data
mkdir -p "$ROOT" artifacts logs

export HELIX_CORPUS_DIR="$ROOT"

echo "=== [1/8] fetch corpus ==="
if [ ! -s "$ROOT/enwiki_head.xml.bz2" ]; then
  curl -sL --range 0-1600000000 -o "$ROOT/enwiki_head.xml.bz2" \
    https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2
fi
ls -la "$ROOT/enwiki_head.xml.bz2"

echo "=== [2/8] fetch teacher model ==="
if [ ! -s "$ROOT/qwen/model.safetensors" ]; then
  mkdir -p "$ROOT/qwen"
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    curl -sL -o "$ROOT/qwen/$f" "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/$f"
  done
  curl -sL -o "$ROOT/qwen/model.safetensors" \
    https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors
fi
ls -la "$ROOT/qwen/model.safetensors"

echo "=== [3/8] stage1: dump -> text ==="
python3 corpus_pipeline.py 1
echo "=== [4/8] stage2: sentences + dedup ==="
python3 corpus_pipeline.py 2
echo "=== [5/8] stage3: definitional extraction ==="
python3 corpus_pipeline.py 3
echo "=== [6/8] stage5: MDL rule induction ==="
python3 corpus_pipeline.py 5
echo "=== [7/8] stage6+6b: 250 MB artifact (hot DNA + cold store + index) ==="
python3 corpus_pipeline.py 6
python3 corpus_pipeline.py 6b
echo "=== [8/8] stage7: held-out evaluation ==="
python3 corpus_pipeline.py 7
echo "=== DONE ==="
