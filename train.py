"""Tiny BERT masked-language-model baseline. Run with `uv run train.py`.

The agent edits this file. Data, evaluation and resource limits live in prepare.py.
This is MLM-only pretraining from random initialization, without NSP or a pooler.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from prepare import (
    DATA_DIR,
    MAX_PARAMS,
    MAX_SEQ_LEN,
    PAD_ID,
    TIME_BUDGET,
    choose_device,
    evaluate_mlm,
    make_dataloader,
    mask_tokens,
    peak_host_rss_mb,
    read_manifest,
    synchronize,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class BertConfig:
    vocab_size: int = 4096
    max_seq_len: int = MAX_SEQ_LEN
    n_layer: int = 2
    n_head: int = 4
    hidden_size: int = 128
    intermediate_size: int = 512
    dropout: float = 0.1


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.n_head:
            raise ValueError("Hidden size must be divisible by head count.")
        self.n_head = config.n_head
        self.head_dim = config.hidden_size // config.n_head
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = config.dropout

    def forward(self, x, attention_mask):
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, length, 3, self.n_head, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask[:, None, None, :].bool(),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SelfAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask):
        x = self.attention_norm(x + self.dropout(self.attention(x, attention_mask)))
        return self.output_norm(x + self.dropout(self.ffn(x)))


class TinyBert(nn.Module):
    """forward returns [number of selected positions, vocabulary size] logits.

    `selected` is a boolean position mask, never the original target token IDs.
    Evaluation labels and loss calculation are owned by prepare.py.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=PAD_ID
        )
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.type_embeddings = nn.Embedding(2, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.mlm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlm_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.apply(self._initialize)
        with torch.no_grad():
            self.word_embeddings.weight[PAD_ID].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, selected):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(positions)
            + self.type_embeddings(torch.zeros_like(input_ids))
        )
        x = self.dropout(self.embedding_norm(x))
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.mlm_norm(F.gelu(self.mlm_dense(x[selected])))
        return F.linear(x, self.word_embeddings.weight, self.mlm_bias)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Run configuration
SEED = 17
THREADS = 4
DEVICE = "auto"  # "auto", "cpu", "mps", or "cuda"

# Model architecture
DEPTH = 2
HIDDEN_SIZE = 128
NUM_HEADS = 4
INTERMEDIATE_SIZE = 512
DROPOUT = 0.1

# Optimization
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.9, 0.999)
WARMUP_RATIO = 0.1
GRAD_CLIP = 1.0

# Batching
DEVICE_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4
MASK_RATE = 0.6

# ---------------------------------------------------------------------------
# Setup: data, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)
device = choose_device(DEVICE)
print(f"Device: {device}")

manifest = read_manifest(DATA_DIR)
config = BertConfig(
    vocab_size=manifest["vocab_size"],
    n_layer=DEPTH,
    n_head=NUM_HEADS,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=INTERMEDIATE_SIZE,
    dropout=DROPOUT,
)
print(f"Model config: {asdict(config)}")

model = TinyBert(config).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Num params: {num_params:,}")
if num_params > MAX_PARAMS:
    raise ValueError(f"Parameter limit exceeded: {num_params} > {MAX_PARAMS}")
if DEVICE_BATCH_SIZE < 1 or GRAD_ACCUM_STEPS < 1:
    raise ValueError("Batch size and accumulation must be positive.")

decay, no_decay = [], []
for parameter in model.parameters():
    (decay if parameter.ndim >= 2 else no_decay).append(parameter)
optimizer = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=LEARNING_RATE,
    betas=ADAM_BETAS,
    eps=1e-8,
)

train_loader = make_dataloader(DATA_DIR, DEVICE_BATCH_SIZE, SEED + 101)
mask_generator = torch.Generator().manual_seed(SEED + 211)
next_ids = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

model.train()
steps, tokens, selected_tokens = 0, 0, 0
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()
synchronize(device)
t_start_training = time.time()

while time.time() - t_start_training < TIME_BUDGET:
    elapsed = time.time() - t_start_training
    progress = min(elapsed / TIME_BUDGET, 1.0)
    if WARMUP_RATIO > 0 and progress < WARMUP_RATIO:
        lr_multiplier = max(progress / WARMUP_RATIO, 1e-3)
    else:
        lr_multiplier = max(0.0, (1.0 - progress) / max(1.0 - WARMUP_RATIO, 1e-8))
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE * lr_multiplier

    batches = []
    for _ in range(GRAD_ACCUM_STEPS):
        batches.append(mask_tokens(next_ids, config.vocab_size, mask_generator, MASK_RATE))
        next_ids = next(train_loader)
    n_targets = sum(int(b["selected"].sum()) for b in batches)

    optimizer.zero_grad(set_to_none=True)
    step_loss = 0.0
    for batch in batches:
        tokens += int(batch["attention_mask"].sum())
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"], batch["selected"])
        targets = batch["targets"][batch["selected"]]
        loss = F.cross_entropy(logits, targets, reduction="sum") / n_targets
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")
        loss.backward()
        step_loss += loss.detach().item()
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
    optimizer.step()
    synchronize(device)

    steps += 1
    selected_tokens += n_targets
    if steps == 1 or steps % 25 == 0:
        print(
            f"step {steps:05d} | train_loss: {step_loss:.6f} | "
            f"elapsed: {time.time() - t_start_training:.1f}s",
            flush=True,
        )

synchronize(device)
training_seconds = time.time() - t_start_training
if steps == 0:
    raise RuntimeError("No optimizer steps completed.")

# ---------------------------------------------------------------------------
# Evaluation (in-process, same weights, fixed val split from prepare.py)
# ---------------------------------------------------------------------------

val_inputs = torch.load(DATA_DIR / "val_inputs.pt", weights_only=True)
val_targets = torch.load(DATA_DIR / "val_targets.pt", weights_only=True)
evaluation = evaluate_mlm(model, val_inputs, val_targets, device)

t_end = time.time()
peak_rss_mb = peak_host_rss_mb()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else None

print()
print("---")
print(f"val_mlm_loss:     {evaluation['mlm_loss']:.6f}")
print(f"val_accuracy:     {evaluation['accuracy']:.6f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_rss_mb:      {peak_rss_mb}")
if peak_vram_mb is not None:
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_tokens_M:   {tokens / 1e6:.3f}")
print(f"selected_tokens_M:{selected_tokens / 1e6:.3f}")
print(f"num_steps:        {steps}")
print(f"num_params_M:     {num_params / 1e6:.3f}")
print(f"depth:            {DEPTH}")
