#!/usr/bin/env bash
#
# run1.sh -- GPT-2-small scale (124M params) training run on FineWeb.
#
#   data:      data/fineweb   (3,677,797,529 GPT-2 BPE tokens, memmapped)
#   model:     12 layers / 12 heads / 768 embd  ~= 124M params
#   schedule:  30000 iters x 32,768 tokens = ~983M tokens, lr 6e-4 cosine -> 6e-5
#   ETA:       ~6.5-7 h on the RTX 4070 (llama-server must be stopped)
#
# Usage (in tmux):
#     tmux new -s igpt
#     ./run1.sh
#     # detach: Ctrl-b d   |   reattach: tmux attach -t igpt
#
# Resume after an interruption (restores optimizer state too):
#     ./run1.sh --init_from resume
#
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-/usr/bin/python3.12}"          # system python 3.12 (torch 2.11+cu130, tiktoken)
DATA_DIR="${DATA_DIR:-data/fineweb}"
OUT_DIR="${OUT_DIR:-out-fineweb-124m}"

# eval can otherwise build a backward graph and OOM; expandable segments avoids
# fragmentation when VRAM is tight.
export PYTORCH_ALLOC_CONF=expandable_segments:True

for f in train.bin val.bin meta.pkl; do
    if [[ ! -f "$DATA_DIR/$f" ]]; then
        echo "ERROR: missing $DATA_DIR/$f" >&2
        exit 1
    fi
done

mkdir -p "$OUT_DIR"

echo "python : $PY"
echo "data   : $DATA_DIR"
echo "out    : $OUT_DIR"
echo "log    : $OUT_DIR/train.log"
echo "extra  : $*"
echo

"$PY" -u train.py \
    --data_dir "$DATA_DIR" \
    --tokenizer bpe \
    --out_dir "$OUT_DIR" \
    --block_size 512 \
    --n_layer 12 \
    --n_head 12 \
    --n_embd 768 \
    --dropout 0.1 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --max_iters 30000 \
    --warmup_iters 1000 \
    --lr_decay_iters 30000 \
    --learning_rate 6e-4 \
    --min_lr 6e-5 \
    --weight_decay 0.1 \
    --beta1 0.9 \
    --beta2 0.95 \
    --grad_clip 1.0 \
    --eval_interval 500 \
    --eval_iters 50 \
    --log_interval 50 \
    "$@" 2>&1 | tee -a "$OUT_DIR/train.log"
