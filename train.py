#!/usr/bin/env python3
"""
igpt -- a minimal, single-file GPT training script in the spirit of nanoGPT.

Everything lives in this one file:

    * data download / preparation (char-level or GPT-2 BPE tokenizer)
    * tokenizers (a tiny char tokenizer, or tiktoken's BPE)
    * the GPT model (decoder-only transformer, weight-tied)
    * the training loop (AdamW, warmup + cosine LR, grad accumulation, AMP, DDP)
    * checkpointing / resume
    * sampling / generation

Quick start
-----------
    python train.py                       # train on tiny shakespeare (char-level)
    python train.py --tokenizer bpe       # same, but with the GPT-2 BPE tokenizer
    python train.py --sample --prompt "ROMEO:" --num_samples 3

Multi-GPU (DDP):
    torchrun --standalone --nproc_per_node=4 train.py

Only requires: torch, numpy (and tiktoken if you want the BPE tokenizer).
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import time
import urllib.request
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# --------------------------------------------------------------------------------------
# Data: download + tokenize into flat binary files
# --------------------------------------------------------------------------------------

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


class CharTokenizer:
    """Maps characters to integers. Vocab is whatever characters appear in the corpus."""

    def __init__(self, text: str | None = None, stoi: dict | None = None, itos: dict | None = None):
        if stoi is not None and itos is not None:
            self.stoi = stoi
            self.itos = {int(k): v for k, v in itos.items()}
        else:
            assert text is not None, "need either `text` or `stoi`/`itos`"
            chars = sorted(set(text))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def state_dict(self) -> dict:
        return {"tokenizer": "char", "stoi": self.stoi, "itos": self.itos}


class BPETokenizer:
    """Thin wrapper around a tiktoken encoding (default: the GPT-2 BPE)."""

    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken  # optional dependency, imported lazily

        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self.eot = self.enc.eot_token

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode(self, s: str) -> list[int]:
        return self.enc.encode_ordinary(s)

    def decode(self, ids) -> str:
        return self.enc.decode([int(i) for i in ids])

    def state_dict(self) -> dict:
        return {
            "tokenizer": "bpe",
            "encoding_name": self.encoding_name,
            "vocab_size": self.vocab_size,
        }


def download_shakespeare(data_dir: str) -> str:
    path = os.path.join(data_dir, "shakespeare", "input.txt")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading tiny shakespeare -> {path}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    return path


def build_tokenizer(tokenizer_name: str, text: str, meta_path: str):
    """Load a cached tokenizer from meta.pkl, or fit a new one on `text` and cache it."""
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("tokenizer") == tokenizer_name:
            if tokenizer_name == "char":
                tok = CharTokenizer(stoi=meta["stoi"], itos=meta["itos"])
            else:
                tok = BPETokenizer(meta.get("encoding_name", "gpt2"))
            return tok, meta

    if tokenizer_name == "char":
        tok = CharTokenizer(text=text)
    elif tokenizer_name == "bpe":
        tok = BPETokenizer("gpt2")
    else:
        raise ValueError(f"unknown tokenizer: {tokenizer_name!r}")

    meta = tok.state_dict()
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    return tok, meta


def prepare(data_dir: str, dataset: str, tokenizer_name: str) -> int:
    """Produce train.bin / val.bin / meta.pkl under data_dir. Returns vocab_size."""
    os.makedirs(data_dir, exist_ok=True)
    meta_path = os.path.join(data_dir, "meta.pkl")

    # 1) locate the raw text
    if dataset == "shakespeare":
        input_file = download_shakespeare(data_dir)
    elif os.path.isfile(dataset):
        input_file = dataset
    else:
        raise FileNotFoundError(f"--dataset must be 'shakespeare' or a path to a text file, got {dataset!r}")

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # 2) tokenize
    tok, meta = build_tokenizer(tokenizer_name, text, meta_path)
    vocab_size = tok.vocab_size
    if vocab_size >= 2**16:
        raise ValueError(f"vocab_size {vocab_size} does not fit in uint16 storage")

    print(f"dataset: {input_file} ({len(text):,} chars)")
    print(f"tokenizer: {tokenizer_name} (vocab_size={vocab_size})")

    if tokenizer_name == "char":
        ids = tok.encode(text)
    else:
        # keep the GPT-2 <|endoftext|> boundaries as explicit separators
        docs = text.split("<|endoftext|>")
        ids = []
        for i, doc in enumerate(docs):
            if i > 0:
                ids.append(tok.eot)
            ids.extend(tok.encode(doc))

    n = len(ids)
    split = int(n * 0.9)
    arr = np.array(ids, dtype=np.uint16)
    arr[:split].tofile(os.path.join(data_dir, "train.bin"))
    arr[split:].tofile(os.path.join(data_dir, "val.bin"))
    print(f"tokens: {n:,} (train {split:,} / val {n - split:,}) -> {data_dir}")

    # 3) sanity check the tokenizer round-trips
    probe = text[:200]
    assert tok.decode(tok.encode(probe)) == probe, "tokenizer round-trip failed"
    return vocab_size


class BinDataset:
    """A flat uint16 token stream, sampled into random (x, y) batches."""

    def __init__(self, path: str, block_size: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(self, batch_size: int, device: str):
        ix = torch.randint(len(self.data) - self.block_size - 1, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix]
        )
        if device.startswith("cuda"):
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


# --------------------------------------------------------------------------------------
# Model: a decoder-only transformer (GPT-2 flavour, pre-LayerNorm)
# --------------------------------------------------------------------------------------


@dataclass
class GPTConfig:
    block_size: int = 256      # maximum context length
    vocab_size: int = 50304    # overridden at runtime from the tokenizer
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True          # use bias in Linear/LayerNorm (GPT-2 style)


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: input embedding and output projection share weights
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"model: {self.get_num_params() / 1e6:.2f}M parameters")

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is None:
            logits = self.lm_head(x[:, [-1], :])  # only the last position, for generation
            return logits, None

        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """AdamW with weight decay applied only to matrices (ndim >= 2)."""
        params = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in params.values() if p.dim() >= 2]
        no_decay = [p for p in params.values() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        n = sum(p.numel() for p in decay) + sum(p.numel() for p in no_decay)
        print(f"optimizer: {len(decay)} decayed / {len(no_decay)} non-decayed tensors ({n:,} params)")
        kwargs = {}
        if device_type == "cuda":
            try:
                torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=True)
                kwargs["fused"] = True
            except (TypeError, RuntimeError):
                pass
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **kwargs)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive sampling; `idx` is (B, T) LongTensor of prompt token ids."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="igpt: single-file GPT training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data / io
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--dataset", type=str, default="shakespeare", help="'shakespeare' or a path to a text file")
    p.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe"])
    p.add_argument("--out_dir", type=str, default="out")
    p.add_argument("--init_from", type=str, default="scratch", choices=["scratch", "resume"])
    p.add_argument("--always_save_checkpoint", action="store_true", help="save even when val loss does not improve")
    p.add_argument("--eval_only", action="store_true", help="evaluate once and exit")

    # model (small default that trains fine on one GPU / CPU)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bias", type=bool, default=True)
    p.add_argument("--compile", action="store_true", help="torch.compile the model (slow first step, faster after)")

    # optimization
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--lr_decay_iters", type=int, default=None, help="defaults to max_iters")
    p.add_argument("--min_lr", type=float, default=1e-4)  # ~= lr/10
    p.add_argument("--decay_lr", type=bool, default=True)

    # eval / logging
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=200)
    p.add_argument("--log_interval", type=int, default=10)

    # system
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=1337)

    # sampling
    p.add_argument("--sample", action="store_true", help="sample from the (best) checkpoint and exit")
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    return p.parse_args()


def pick_device(arg: str) -> tuple[str, str]:
    if arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = arg
    device_type = "cuda" if device.startswith("cuda") else device
    return device, device_type


def pick_dtype(arg: str, device_type: str) -> str:
    if arg != "auto":
        return arg
    if device_type == "cuda":
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if device_type == "mps":
        return "float16"
    return "float32"


def make_grad_scaler(dtype: str, device_type: str):
    enabled = device_type == "cuda" and dtype == "float16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def unwrap(model):
    m = model
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if hasattr(m, "module"):
        m = m.module
    return m


def load_tokenizer_from_meta(data_dir: str):
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    if meta["tokenizer"] == "char":
        return CharTokenizer(stoi=meta["stoi"], itos=meta["itos"])
    return BPETokenizer(meta.get("encoding_name", "gpt2"))


def sample_and_print(raw_model, tokenizer, args, device):
    raw_model.eval()
    for i in range(args.num_samples):
        prompt_ids = tokenizer.encode(args.prompt)
        if len(prompt_ids) == 0:
            prompt_ids = [0]
        idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
        out = raw_model.generate(
            idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k
        )
        text = tokenizer.decode(out[0].tolist())
        print(f"\n{'=' * 30} sample {i + 1} {'=' * 30}\n{text}")


def main():
    args = parse_args()

    # ---- distributed setup ----
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.is_available():
            device = f"cuda:{local_rank}"
            torch.cuda.set_device(device)
        else:
            device = "cpu"
        device_type = "cuda" if device.startswith("cuda") else "cpu"
        master_process = rank == 0
    else:
        device, device_type = pick_device(args.device)
        rank = 0
        world_size = 1
        master_process = True

    dtype = pick_dtype(args.dtype, device_type)
    if not master_process:
        pass  # keep prints quiet on non-master ranks
    else:
        print(f"device={device} dtype={dtype} world_size={world_size}")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    ctx = torch.amp.autocast(device_type=device_type, dtype=getattr(torch, dtype), enabled=(device_type != "cpu"))
    scaler = make_grad_scaler(dtype, device_type)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")

    # ---- sample-only mode: no data needed, just checkpoint + tokenizer ----
    if args.sample:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"no checkpoint at {ckpt_path}; train first")
        tokenizer = load_tokenizer_from_meta(args.data_dir)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model = GPT(GPTConfig(**ckpt["config"])).to(device)
        raw_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()})
        if raw_model.config.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"checkpoint vocab_size ({raw_model.config.vocab_size}) != tokenizer vocab_size "
                f"({tokenizer.vocab_size}); did you mix --data_dir/--out_dir from different runs?"
            )
        if master_process:
            print(f"loaded {ckpt_path} (iter {ckpt['iter_num']}, val loss {ckpt['best_val_loss']:.4f})")
            sample_and_print(raw_model, tokenizer, args, device)
        return

    # ---- data (re-tokenize only if the cache is missing or for another tokenizer) ----
    cached = all(
        os.path.exists(os.path.join(args.data_dir, f))
        for f in ("train.bin", "val.bin", "meta.pkl")
    )
    if cached:
        with open(os.path.join(args.data_dir, "meta.pkl"), "rb") as f:
            cached = pickle.load(f).get("tokenizer") == args.tokenizer
    if not cached:
        vocab_size = prepare(args.data_dir, args.dataset, args.tokenizer)
    else:
        vocab_size = load_tokenizer_from_meta(args.data_dir).vocab_size

    tokenizer = load_tokenizer_from_meta(args.data_dir)
    train_data = BinDataset(os.path.join(args.data_dir, "train.bin"), args.block_size)
    val_data = BinDataset(os.path.join(args.data_dir, "val.bin"), args.block_size)
    if master_process:
        print(f"data: {len(train_data):,} train tokens / {len(val_data):,} val tokens")

    # ---- model ----
    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )
    if args.init_from == "resume" and os.path.exists(ckpt_path):
        if master_process:
            print(f"resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = GPTConfig(**ckpt["config"])
        raw_model = GPT(config)
        state = ckpt["model"]
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        raw_model.load_state_dict(state)
        iter_num = ckpt["iter_num"]
        best_val_loss = ckpt["best_val_loss"]
    else:
        raw_model = GPT(config)
        iter_num = 0
        best_val_loss = float("inf")

    raw_model.to(device)

    # optimizer
    optimizer = raw_model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
    )
    if args.init_from == "resume" and os.path.exists(ckpt_path):
        optimizer.load_state_dict(ckpt["optimizer"])

    model = raw_model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(device.split(":")[1])] if device_type == "cuda" else None)
    if args.compile:
        if master_process:
            print("compiling model ...")
        model = torch.compile(model)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split, ds in (("train", train_data), ("val", val_data)):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                x, y = ds.get_batch(args.batch_size, device)
                with ctx:
                    _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # ---- training loop ----
    lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters is not None else args.max_iters
    tokens_per_iter = args.batch_size * args.gradient_accumulation_steps * args.block_size * world_size
    if master_process:
        print(
            f"training: {args.max_iters} iters, {tokens_per_iter:,} tokens/iter, "
            f"~{tokens_per_iter * args.max_iters / 1e6:.1f}M tokens total"
        )
        if args.eval_only:
            losses = estimate_loss()
            print(f"eval: train {losses['train']:.4f} | val {losses['val']:.4f}")
            return

    x, y = train_data.get_batch(args.batch_size, device)
    t0 = time.time()
    running_mfu = 0.0

    while iter_num < args.max_iters:
        # set learning rate for this step
        lr = get_lr(iter_num, args.warmup_iters, lr_decay_iters, args.learning_rate, args.min_lr) if args.decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # periodic eval + checkpoint
        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
            losses = estimate_loss()
            if master_process:
                print(
                    f"iter {iter_num}: train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e}"
                )
            if losses["val"] < best_val_loss or args.always_save_checkpoint:
                best_val_loss = min(best_val_loss, losses["val"])
                if master_process and iter_num > 0:
                    torch.save(
                        {
                            "model": unwrap(model).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "config": asdict(config),
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                        },
                        ckpt_path,
                    )
                    print(f"  saved checkpoint (val loss {losses['val']:.4f})")

        # gradient accumulation
        for micro_step in range(args.gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = micro_step == args.gradient_accumulation_steps - 1
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.gradient_accumulation_steps
            x, y = train_data.get_batch(args.batch_size, device)
            scaler.scale(loss).backward()

        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.log_interval == 0 and master_process:
            lossf = loss.item() * args.gradient_accumulation_steps
            if iter_num >= 5 and device_type == "cuda":
                mfu = raw_model.get_num_params() * 6 * tokens_per_iter / (dt * 1e12)
                running_mfu = mfu if running_mfu == 0.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f} | time {dt * 1000:.0f}ms | mfu {running_mfu * 100:.1f}%")

        iter_num += 1

    if ddp:
        torch.distributed.destroy_process_group()

    # final save: keep the last state, but do NOT clobber the best-val checkpoint
    if master_process:
        last_path = os.path.join(args.out_dir, "ckpt_last.pt")
        torch.save(
            {
                "model": unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(config),
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            },
            last_path,
        )
        print(f"training done; final state at {last_path} | best-val ckpt at {ckpt_path}")
        sample_and_print(unwrap(model), tokenizer, args, device)


if __name__ == "__main__":
    main()
