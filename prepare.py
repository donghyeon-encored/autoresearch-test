"""Fixed data and evaluation contract. Run once with `uv run prepare.py`.

This file is frozen during an experiment campaign. Only train.py is editable.
No pretrained weights, runtime text tokenization, or GPT-style shifted targets.
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
PROTOCOL_VERSION = 1
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID = range(5)
REPO_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"

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


def make_eval_views(ids: torch.Tensor, vocab_size: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ids), generator=gen)[:EVAL_CHUNKS]
    rows = ids[indices]
    views = [mask_tokens(rows, vocab_size, gen, EVAL_MASK_RATE) for _ in range(EVAL_VIEWS)]
    return {key: torch.cat([v[key] for v in views]) for key in views[0]}


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
    stats = {}
    for split, docs in documents.items():
        rows, unknown, total = [], 0, 0
        for text in docs:
            tokens = tokenizer.encode(text).ids
            unknown += tokens.count(UNK_ID)
            total += len(tokens)
            for start in range(0, len(tokens), MAX_SEQ_LEN - 2):
                body = tokens[start : start + MAX_SEQ_LEN - 2]
                if not any(t >= len(SPECIAL_TOKENS) for t in body):
                    continue
                row = [CLS_ID] + body + [SEP_ID]
                rows.append(row + [PAD_ID] * (MAX_SEQ_LEN - len(row)))
        if not rows:
            raise ValueError(f"No usable token chunks in {split}.")
        ids = torch.tensor(rows, dtype=torch.int32)
        stats[split] = {
            "documents": len(docs),
            "chunks": len(rows),
            "tokens": total,
            "unknown_fraction": unknown / max(total, 1),
        }
        if split == "train":
            torch.save(ids, output / "train.pt")
        else:
            views = make_eval_views(ids.long(), vocab_size, 2029 if split == "val" else 3037)
            torch.save(
                {k: views[k] for k in ("input_ids", "attention_mask", "selected")},
                output / f"{split}_inputs.pt",
            )
            torch.save(
                {k: views[k] for k in ("targets", "mask_only")}, output / f"{split}_targets.pt"
            )
            stats[split]["evaluation_rows"] = len(views["input_ids"])
            stats[split]["evaluation_targets"] = int(views["selected"].sum())
    names = [
        "tokenizer.json",
        "train.pt",
        "val_inputs.pt",
        "val_targets.pt",
        "test_inputs.pt",
        "test_targets.pt",
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
    """Infinite shuffled training chunks on CPU. Never reads held-out data."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    ids = torch.load(data_dir / "train.pt", map_location="cpu", weights_only=True).long()
    if len(ids) == 0:
        raise ValueError("No training chunks.")
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
        yield ids[torch.cat(pieces)]


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
def evaluate_mlm(
    model, inputs: dict, answers: dict, device: str, batch_size: int = EVAL_BATCH_SIZE
) -> dict:
    """Compute target-weighted CE against the live model. Never pass labels to forward."""
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be positive.")
    was_training = model.training
    model.eval()
    nll, correct, count, mask_nll, mask_correct, mask_count = 0.0, 0, 0, 0.0, 0, 0
    for start in range(0, len(inputs["input_ids"]), batch_size):
        stop = start + batch_size
        batch = {k: v[start:stop].to(device) for k, v in inputs.items()}
        selected = batch["selected"]
        targets = answers["targets"][start:stop].to(device)[selected].long()
        mask_only = answers["mask_only"][start:stop].to(device)[selected]
        logits = model(batch["input_ids"].long(), batch["attention_mask"], selected).float()
        if logits.shape != (targets.numel(), model.config.vocab_size):
            raise ValueError("Model returned incorrectly shaped logits.")
        if not torch.isfinite(logits).all():
            raise ValueError("Model returned non-finite logits.")
        losses = F.cross_entropy(logits, targets, reduction="none")
        matches = logits.argmax(-1) == targets
        # Sum on CPU in float64 so different batch sizes do not change weighting.
        nll += losses.cpu().double().sum().item()
        correct += matches.sum().item()
        count += targets.numel()
        mask_nll += losses[mask_only].cpu().double().sum().item()
        mask_correct += matches[mask_only].sum().item()
        mask_count += mask_only.sum().item()
    if was_training:
        model.train()
    if count == 0:
        raise ValueError("No evaluation targets.")
    return {
        "mlm_loss": nll / count,
        "accuracy": correct / count,
        "targets": count,
        "mask_only_loss": mask_nll / mask_count if mask_count else None,
        "mask_only_accuracy": mask_correct / mask_count if mask_count else None,
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
