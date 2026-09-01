"""Fixed data and evaluation contract. Run once with `uv run prepare.py`.

Multi-task pretraining contract: masked-language-modeling (MLM) + causal
next-token modeling (LM) + next-segment classification (NSP-style), all
sharing one encoder trunk. This file is frozen during an experiment
campaign. Only train.py is editable. Models must implement forward_mlm,
forward_lm, and forward_nsp -- see evaluate_multitask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / ".data" / "bert"
MAX_SEQ_LEN = 128
VOCAB_SIZE = 4096
TIME_BUDGET = 120.0  # training time budget in seconds (wall clock, training only)
MAX_PARAMS = 3_000_000
EVAL_CHUNKS = 512
EVAL_VIEWS = 2
EVAL_BATCH_SIZE = 16
EVAL_MASK_RATE = 0.15
PROTOCOL_VERSION = 2
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID = range(5)
REPO_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
PAIR_SEEDS = {"train": 1013, "val": 2029, "test": 3037}
NSP_POSITIVE_PROB = 0.5

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def read_manifest(data_dir: Path, verify: bool = False) -> dict:
    path = data_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Prepared data missing: {path}. Run `uv run prepare.py` first.")
    manifest = json.loads(path.read_text())
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("Data protocol differs; prepare a new data directory.")
    if manifest["sequence_length"] != MAX_SEQ_LEN:
        raise ValueError("Sequence length differs from the frozen protocol.")
    if verify:
        for name, expected in manifest["files"].items():
            if sha256(data_dir / name) != expected:
                raise ValueError(f"Data integrity failure: {name}")
    return manifest


# ---------------------------------------------------------------------------
# Data download + artifact building
# ---------------------------------------------------------------------------


def documents_from_lines(lines: list[str]) -> list[str]:
    """WikiText top-level '= title =' starts a document; nested headings do not."""
    documents, current = [], []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"=\s*[^=]+?\s*=", line) and current:
            documents.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        documents.append("\n".join(current))
    return documents


def remove_train_duplicates(documents: dict[str, list[str]]) -> tuple[dict, list[str]]:
    def digest(text):
        return hashlib.sha256(text.encode()).hexdigest()

    held_out = {digest(d) for split in ("val", "test") for d in documents[split]}
    removed = [digest(d) for d in documents["train"] if digest(d) in held_out]
    result = dict(documents)
    result["train"] = [d for d in documents["train"] if digest(d) not in held_out]
    return result, removed


def mask_tokens(
    ids: torch.Tensor, vocab_size: int, generator: torch.Generator, rate: float = 0.15
) -> dict[str, torch.Tensor]:
    """CPU corruption, with one or more targets per nonempty row and separate RNG."""
    if not 0 < rate <= 1 or vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError("Invalid masking rate or vocabulary.")
    eligible = ids >= len(SPECIAL_TOKENS)
    if not eligible.any(dim=1).all():
        raise ValueError("Every row must contain a non-special token.")
    selected = (torch.rand(ids.shape, generator=generator) < rate) & eligible
    for row in (~selected.any(dim=1)).nonzero().flatten().tolist():
        positions = eligible[row].nonzero().flatten()
        pick = torch.randint(len(positions), (1,), generator=generator).item()
        selected[row, positions[pick]] = True
    draws = torch.rand(ids.shape, generator=generator)
    masked = selected & (draws < 0.8)
    random_replacement = selected & (draws >= 0.8) & (draws < 0.9)
    corrupted = ids.clone()
    corrupted[masked] = MASK_ID
    random_ids = torch.randint(len(SPECIAL_TOKENS), vocab_size, ids.shape, generator=generator)
    corrupted[random_replacement] = random_ids[random_replacement]
    targets = ids.clone()
    targets[~selected] = -100
    return {
        "input_ids": corrupted,
        "attention_mask": ids != PAD_ID,
        "selected": selected,
        "targets": targets,
        "mask_only": masked,
    }


def _chunk_document(tokens: list[int], seg_len: int) -> list[list[int]]:
    chunks = []
    for start in range(0, len(tokens), seg_len):
        piece = tokens[start : start + seg_len]
        if any(t >= len(SPECIAL_TOKENS) for t in piece):
            chunks.append(piece)
    return chunks


def _build_pairs(doc_chunks: list[list[list[int]]], seed: int) -> tuple[list, list, list]:
    """Adjacent chunks within a document are IsNext; otherwise B is a random pool chunk."""
    pool = [chunk for chunks in doc_chunks for chunk in chunks]
    gen = torch.Generator().manual_seed(seed)
    rows, seg_id_rows, is_next = [], [], []
    for chunks in doc_chunks:
        for i in range(len(chunks) - 1):
            a, b_true = chunks[i], chunks[i + 1]
            positive = torch.rand((), generator=gen).item() < NSP_POSITIVE_PROB
            b = b_true if positive else pool[torch.randint(len(pool), (1,), generator=gen).item()]
            body = a + [SEP_ID] + b
            row = [CLS_ID] + body + [SEP_ID]
            seg = [0] * (len(a) + 2) + [1] * (len(b) + 1)
            pad = MAX_SEQ_LEN - len(row)
            rows.append(row + [PAD_ID] * pad)
            seg_id_rows.append(seg + [0] * pad)
            is_next.append(int(positive))
    return rows, seg_id_rows, is_next


def build_artifacts(documents: dict[str, list[str]], output: Path, source: dict) -> dict:
    """Also used by offline tests, with a clearly separate fixture data directory."""
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    output.mkdir(parents=True, exist_ok=True)
    documents, removed = remove_train_duplicates(documents)
    if any(not documents[s] for s in ("train", "val", "test")):
        raise ValueError("Each split needs at least one document after duplicate removal.")
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.train_from_iterator(
        documents["train"],
        trainers.WordPieceTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS),
    )
    assert [tokenizer.token_to_id(t) for t in SPECIAL_TOKENS] == list(range(5))
    tokenizer.save(str(output / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()

    seg_len = (MAX_SEQ_LEN - 3) // 2
    stats = {}
    for split, docs in documents.items():
        doc_chunks, unknown, total = [], 0, 0
        for text in docs:
            tokens = tokenizer.encode(text).ids
            unknown += tokens.count(UNK_ID)
            total += len(tokens)
            chunks = _chunk_document(tokens, seg_len)
            if len(chunks) >= 2:
                doc_chunks.append(chunks)
        if not doc_chunks:
            raise ValueError(f"No document with adjacent chunk pairs in {split}.")

        rows, seg_id_rows, is_next = _build_pairs(doc_chunks, PAIR_SEEDS[split])
        ids = torch.tensor(rows, dtype=torch.int32)
        segment_ids = torch.tensor(seg_id_rows, dtype=torch.int32)
        is_next_t = torch.tensor(is_next, dtype=torch.int64)

        stats[split] = {
            "documents": len(docs),
            "documents_with_pairs": len(doc_chunks),
            "pairs": len(rows),
            "tokens": total,
            "unknown_fraction": unknown / max(total, 1),
            "positive_fraction": sum(is_next) / len(is_next),
        }

        if split == "train":
            torch.save(
                {"input_ids": ids, "segment_ids": segment_ids, "is_next": is_next_t},
                output / "train.pt",
            )
        else:
            eval_seed = PAIR_SEEDS[split]
            perm_gen = torch.Generator().manual_seed(eval_seed)
            indices = torch.randperm(len(ids), generator=perm_gen)[:EVAL_CHUNKS]
            clean_ids = ids[indices].long()
            clean_segments = segment_ids[indices].long()
            clean_is_next = is_next_t[indices]
            torch.save(
                {
                    "input_ids": clean_ids,
                    "attention_mask": clean_ids != PAD_ID,
                    "segment_ids": clean_segments,
                    "is_next": clean_is_next,
                },
                output / f"{split}_clean.pt",
            )
            mlm_gen = torch.Generator().manual_seed(eval_seed + 1)
            views = [
                mask_tokens(clean_ids, vocab_size, mlm_gen, EVAL_MASK_RATE) for _ in range(EVAL_VIEWS)
            ]
            torch.save(
                {
                    "input_ids": torch.cat([v["input_ids"] for v in views]),
                    "attention_mask": torch.cat([v["attention_mask"] for v in views]),
                    "selected": torch.cat([v["selected"] for v in views]),
                    "segment_ids": clean_segments.repeat(EVAL_VIEWS, 1),
                },
                output / f"{split}_mlm_inputs.pt",
            )
            torch.save(
                {
                    "targets": torch.cat([v["targets"] for v in views]),
                    "mask_only": torch.cat([v["mask_only"] for v in views]),
                },
                output / f"{split}_mlm_targets.pt",
            )
            stats[split]["evaluation_rows"] = len(clean_ids)
            stats[split]["evaluation_mlm_targets"] = int(
                torch.cat([v["selected"] for v in views]).sum()
            )

    names = [
        "tokenizer.json",
        "train.pt",
        "val_clean.pt",
        "val_mlm_inputs.pt",
        "val_mlm_targets.pt",
        "test_clean.pt",
        "test_mlm_inputs.pt",
        "test_mlm_targets.pt",
    ]
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "source": source,
        "sequence_length": MAX_SEQ_LEN,
        "vocab_size": vocab_size,
        "eval_mask_rate": EVAL_MASK_RATE,
        "eval_views": EVAL_VIEWS,
        "special_tokens": SPECIAL_TOKENS,
        "stats": stats,
        "removed_train_document_hashes": removed,
        "files": {name: sha256(output / name) for name in names},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def prepare_data(data_dir: Path, revision: str = "main") -> dict:
    if (data_dir / "manifest.json").exists():
        return read_manifest(data_dir, verify=True)
    # Pin the resolved commit even when the user supplied a moving branch name.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    import httpx
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(REPO_ID, revision=revision)
    resolved_revision = info.sha
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset", revision=resolved_revision)
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    cache = data_dir.parent / "hf-cache"
    documents, raw_hashes = {}, {}
    for remote, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        names = sorted(
            n
            for n in files
            if n.startswith(f"{DATASET_CONFIG}/{remote}-") and n.endswith(".parquet")
        )
        if not names:
            raise ValueError(f"No {remote} shards at pinned dataset revision.")
        lines = []
        for name in names:
            print(f"Downloading {name}", flush=True)
            for attempt in range(3):
                try:
                    path = Path(
                        hf_hub_download(
                            REPO_ID,
                            name,
                            repo_type="dataset",
                            revision=resolved_revision,
                            cache_dir=cache,
                        )
                    )
                    break
                except (httpx.TransportError, OSError):
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
            raw_hashes[name] = sha256(path)
            lines.extend(pq.read_table(path, columns=["text"])["text"].to_pylist())
        documents[split] = documents_from_lines(lines)
    source = {
        "repo": REPO_ID,
        "config": DATASET_CONFIG,
        "revision": resolved_revision,
        "raw_sha256": raw_hashes,
    }
    # Publish only after all artifacts were prepared successfully; never overwrite data.
    with tempfile.TemporaryDirectory(prefix="bert-prepare-", dir=data_dir.parent) as temporary:
        stage = Path(temporary) / "artifacts"
        manifest = build_artifacts(documents, stage, source)
        if data_dir.exists():
            raise FileExistsError(
                f"Incomplete/existing directory: {data_dir}; choose a new --data-dir."
            )
        stage.rename(data_dir)
    return manifest


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------


def choose_device(requested: str) -> str:
    if requested == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested not in ("cpu", "mps", "cuda"):
        raise ValueError(f"Unknown device: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable.")
    return requested


def make_dataloader(data_dir: Path, batch_size: int, seed: int):
    """Infinite shuffled training pairs on CPU. Never reads held-out data."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    data = torch.load(data_dir / "train.pt", map_location="cpu", weights_only=True)
    ids = data["input_ids"].long()
    segment_ids = data["segment_ids"].long()
    is_next = data["is_next"].long()
    if len(ids) == 0:
        raise ValueError("No training pairs.")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ids), generator=generator)
    position = 0
    while True:
        pieces, remaining = [], batch_size
        while remaining:
            take = min(remaining, len(ids) - position)
            pieces.append(order[position : position + take])
            position += take
            remaining -= take
            if position == len(ids):
                order = torch.randperm(len(ids), generator=generator)
                position = 0
        idx = torch.cat(pieces)
        yield {"input_ids": ids[idx], "segment_ids": segment_ids[idx], "is_next": is_next[idx]}


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def peak_host_rss_mb() -> float | None:
    """OS-reported peak for this process; unavailable platforms return null, not zero."""
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


@torch.inference_mode()
def evaluate_multitask(
    model, clean: dict, mlm_inputs: dict, mlm_targets: dict, device: str,
    batch_size: int = EVAL_BATCH_SIZE,
) -> dict:
    """Compute MLM + causal-LM + NSP losses against the live model, and a fixed composite.

    `clean` holds unmasked paired rows (for LM + NSP); `mlm_inputs`/`mlm_targets` hold the
    precomputed masked views (for MLM). Never pass labels into any forward_* call.
    """
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be positive.")
    was_training = model.training
    model.eval()
    vocab_size = model.config.vocab_size

    mlm_nll, mlm_correct, mlm_count = 0.0, 0, 0
    for start in range(0, len(mlm_inputs["input_ids"]), batch_size):
        stop = start + batch_size
        batch = {k: v[start:stop].to(device) for k, v in mlm_inputs.items()}
        selected = batch["selected"]
        targets = mlm_targets["targets"][start:stop].to(device)[selected].long()
        logits = model.forward_mlm(
            batch["input_ids"].long(), batch["attention_mask"], batch["segment_ids"].long(), selected
        ).float()
        if logits.shape != (targets.numel(), vocab_size):
            raise ValueError("forward_mlm returned incorrectly shaped logits.")
        if not torch.isfinite(logits).all():
            raise ValueError("forward_mlm returned non-finite logits.")
        losses = F.cross_entropy(logits, targets, reduction="none")
        mlm_nll += losses.cpu().double().sum().item()
        mlm_correct += (logits.argmax(-1) == targets).sum().item()
        mlm_count += targets.numel()

    lm_nll, lm_correct, lm_count = 0.0, 0, 0
    nsp_nll, nsp_correct, nsp_count = 0.0, 0, 0
    for start in range(0, len(clean["input_ids"]), batch_size):
        stop = start + batch_size
        batch = {k: v[start:stop].to(device) for k, v in clean.items()}
        ids = batch["input_ids"].long()
        segment_ids = batch["segment_ids"].long()

        lm_logits = model.forward_lm(ids, batch["attention_mask"], segment_ids).float()
        lm_targets = ids[:, 1:]
        lm_valid = batch["attention_mask"][:, 1:]
        if lm_logits.shape != (ids.shape[0], ids.shape[1] - 1, vocab_size):
            raise ValueError("forward_lm returned incorrectly shaped logits.")
        if not torch.isfinite(lm_logits).all():
            raise ValueError("forward_lm returned non-finite logits.")
        lm_losses = F.cross_entropy(
            lm_logits.reshape(-1, vocab_size), lm_targets.reshape(-1), reduction="none"
        ).reshape(lm_targets.shape)
        lm_nll += lm_losses[lm_valid].cpu().double().sum().item()
        lm_correct += (lm_logits.argmax(-1) == lm_targets)[lm_valid].sum().item()
        lm_count += lm_valid.sum().item()

        nsp_logits = model.forward_nsp(ids, batch["attention_mask"], segment_ids).float()
        nsp_targets = batch["is_next"].to(device).long()
        if nsp_logits.shape != (nsp_targets.numel(), 2):
            raise ValueError("forward_nsp returned incorrectly shaped logits.")
        if not torch.isfinite(nsp_logits).all():
            raise ValueError("forward_nsp returned non-finite logits.")
        nsp_losses = F.cross_entropy(nsp_logits, nsp_targets, reduction="none")
        nsp_nll += nsp_losses.cpu().double().sum().item()
        nsp_correct += (nsp_logits.argmax(-1) == nsp_targets).sum().item()
        nsp_count += nsp_targets.numel()

    if was_training:
        model.train()
    if mlm_count == 0 or lm_count == 0 or nsp_count == 0:
        raise ValueError("No evaluation targets for one or more tasks.")

    mlm_loss = mlm_nll / mlm_count
    lm_loss = lm_nll / lm_count
    nsp_loss = nsp_nll / nsp_count
    return {
        "mlm_loss": mlm_loss,
        "mlm_accuracy": mlm_correct / mlm_count,
        "lm_loss": lm_loss,
        "lm_accuracy": lm_correct / lm_count,
        "nsp_loss": nsp_loss,
        "nsp_accuracy": nsp_correct / nsp_count,
        # Fixed composite score -- the single ground-truth ranking metric. MLM and LM are
        # both vocab_size-way losses (comparable scale); NSP is binary (~0.69 nats at
        # chance) so it is weighted down to act as an auxiliary signal, not a dominant one.
        "composite_loss": mlm_loss + lm_loss + 0.5 * nsp_loss,
    }


# ---------------------------------------------------------------------------
# Main (one-time data prep)
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    result = (
        read_manifest(data_dir, verify=True)
        if args.verify
        else prepare_data(data_dir, args.revision)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
