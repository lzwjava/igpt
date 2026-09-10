# igpt

> **Built with help from the `deepseek-v4-flash` model.**

A minimal, **single-file** GPT training script in the spirit of
[nanoGPT](https://github.com/karpathy/nanoGPT). Everything — data preparation,
tokenizer, model, training loop, checkpointing and sampling — lives in one file:
[`train.py`](train.py).

There is no config file, no package to install, and no `model.py` / `data.py` /
`tokenizer.py` split. Copy the file, run it, read it.

```
igpt/
├── train.py      # the whole project
├── README.md
└── .gitignore
```

## Install

Only `torch` and `numpy` are required:

```bash
pip install torch numpy
# optional, for --tokenizer bpe (GPT-2 BPE):
pip install tiktoken
```

## Quick start

Train a small character-level GPT on tiny shakespeare (downloads ~1.1 MB on first run):

```bash
python train.py
```

Sample from the best checkpoint:

```bash
python train.py --sample --prompt "ROMEO:" --num_samples 3 --max_new_tokens 500
```

Use the GPT-2 BPE tokenizer instead of characters:

```bash
python train.py --tokenizer bpe
```

Turn on mixture-of-experts feed-forward blocks (top-2 of 8 experts per block):

```bash
python train.py --moe
```

Train on your own text file:

```bash
python train.py --dataset /path/to/corpus.txt --tokenizer bpe
```

Multi-GPU / multi-node with DDP:

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

Resume from the last checkpoint (also restores the optimizer state):

```bash
python train.py --init_from resume
```

Evaluate without training:

```bash
python train.py --eval_only
```

On a single modern GPU the default config reaches a val loss around **~1.9**
after ~5k iterations and produces fairly Shakespeare-looking text.

## How it works

### 1. Data (`prepare`)
`--dataset shakespeare` downloads tiny shakespeare; any other value is treated as a
path to a text file. The text is tokenized once into flat `uint16` arrays
`data/train.bin` and `data/val.bin` (90/10 split) plus a cached `data/meta.pkl`.
On later runs the cache is reused, so tokenization happens only once.

### 2. Tokenizer
Two are built in:

| `--tokenizer` | Class           | Vocab | Notes                                            |
| ------------- | --------------- | ----- | ------------------------------------------------ |
| `char`        | `CharTokenizer` | ~65   | character-level, fitted on the corpus, no deps   |
| `bpe`         | `BPETokenizer`  | 50257 | GPT-2 BPE via `tiktoken`, `<|endoftext|>` splits |

Both expose the same `encode` / `decode` / `vocab_size` interface.

### 3. Model (`GPT`)
A decoder-only transformer, GPT-2 flavour:

- learned token + position embeddings (no RoPE/ALiBi)
- pre-LayerNorm blocks: causal self-attention + MLP (GELU, 4x expansion)
- `F.scaled_dot_product_attention` with `is_causal=True` (Flash Attention when available)
- weight tying between the token embedding and the LM head
- GPT-2 style init, with residual projections scaled by `1/sqrt(2 * n_layer)`
- optional **mixture-of-experts** feed-forward blocks (`--moe`): a learned router sends
  each token to the top-k of `--n_experts` experts (Mixtral-style routing) with a
  Switch-Transformer-style load-balancing auxiliary loss

Defaults are deliberately small (`n_layer=6, n_head=6, n_embd=384, block_size=256`,
≈10.7M parameters) so the whole thing trains comfortably on one GPU or even a CPU.

### 4. Training
- AdamW with weight decay on matrices only (no decay on biases / LayerNorms)
- warmup + cosine decay learning-rate schedule
- gradient accumulation (`--gradient_accumulation_steps`)
- mixed precision: `bfloat16` on modern GPUs, `float16` + `GradScaler` otherwise
- gradient clipping, periodic eval, best-val-loss checkpointing
- with `--moe`, a load-balancing auxiliary loss (coefficient `--moe_aux_loss_coef`)
  is added to the cross-entropy loss to keep the router from collapsing onto a few experts
- optional `torch.compile` (`--compile`)
- optional DDP via `torchrun` (no code changes needed)

## Common flags

| Flag                             | Default       | Description                                   |
| -------------------------------- | ------------- | --------------------------------------------- |
| `--dataset`                      | `shakespeare` | `shakespeare` or a path to a text file         |
| `--tokenizer`                    | `char`        | `char` or `bpe`                               |
| `--data_dir` / `--out_dir`       | `data` / `out`| tokenized data / checkpoints                  |
| `--block_size`                   | `256`         | context length                                |
| `--n_layer` / `--n_head` / `--n_embd` | `6` / `6` / `384` | model size                          |
| `--moe`                          | off           | mixture-of-experts feed-forward blocks         |
| `--n_experts` / `--n_experts_active` | `8` / `2` | MoE: experts per block / top-k routed per token |
| `--moe_expert_dim`               | `4`           | MoE: expert hidden dim, as a multiple of `n_embd` |
| `--moe_aux_loss_coef`            | `0.01`        | MoE: load-balancing auxiliary loss coefficient  |
| `--dropout`                      | `0.0`         | raise it (e.g. `0.1`) on small datasets        |
| `--batch_size`                   | `64`          | micro-batch size                              |
| `--gradient_accumulation_steps`  | `1`           | effective batch = batch_size * this * world   |
| `--max_iters`                    | `5000`        | training iterations                           |
| `--learning_rate`                | `1e-3`        | peak LR (warmup + cosine to `--min_lr`)        |
| `--eval_interval` / `--eval_iters` | `250` / `200` | eval cadence / batches per eval             |
| `--compile`                      | off           | `torch.compile` the model                     |
| `--device` / `--dtype`           | `auto`        | `auto`/`cpu`/`cuda`/`mps`, `auto`/`float32`/`bfloat16`/`float16` |
| `--sample`                       | off           | sample from `out/ckpt.pt` and exit            |
| `--prompt`                       | `"\n"`        | sampling prompt                               |
| `--temperature` / `--top_k`      | `0.8` / `200` | sampling controls                             |

Run `python train.py --help` for the full list.

## Scaling up

The defaults are a starting point, not a limit. A rough recipe for a bigger run:

```bash
python train.py \
  --n_layer 12 --n_head 12 --n_embd 768 --block_size 1024 \
  --batch_size 12 --gradient_accumulation_steps 40 \
  --max_iters 60000 --learning_rate 6e-4 --warmup_iters 2000 \
  --dropout 0.1 --compile
```

Tips:

- keep `n_embd % n_head == 0`
- with `--moe`, total parameters grow with `--n_experts` but only top-k experts run per
  token; the script prints both total and active parameter counts
- `block_size` cannot exceed what the data supports; with `char` on shakespeare 256–512 is plenty
- if you hit OOM, lower `batch_size` first, then `block_size`
- `--gradient_accumulation_steps` raises the effective batch size without more memory

## Notes and limitations

- Tokenization reads the whole corpus into memory. For multi-GB corpora, pre-tokenize
  once and point `--data_dir` at the resulting `train.bin` / `val.bin` / `meta.pkl`.
- Only decoder-only (`GPT`) models are implemented; no fine-tuning-from-pretrained-weights
  loader or HF conversion.
- `--tokenizer bpe` uses the GPT-2 vocab (50257); it is a fixed-size vocabulary, so the
  model is larger than the char-level one for the same `n_embd`.
- The MoE dispatch runs each expert as a dense batch (a Python loop over experts), which
  is simple and correct but slower than a fused all-to-all kernel; fine at this scale,
  not a sparse-MoE serving system.

## Acknowledgements

Directly inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and
[minGPT](https://github.com/karpathy/minGPT); the model follows the GPT-2
architecture from *Language Models are Unsupervised Multitask Learners*.

## License

MIT.
