# autoresearch

This is an experiment to have an LLM conduct its own research on a small
multi-task BERT-style pretraining model. Start from the provided working
baseline and discover improvements yourself. No search recipe or preferred
solution is supplied.

The model is trained on three objectives sharing one encoder trunk:

1. **MLM** -- masked-language-modeling, same as classic BERT.
2. **LM** -- causal next-token modeling over the same tokens.
3. **NSP** -- binary next-segment classification: given `[CLS] A [SEP] B [SEP]`,
   predict whether `B` truly follows `A` in the source document or is a random
   segment.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `.data/bert/` contains `manifest.json` and the `.pt` tensors. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single device (CPU, MPS, or CUDA). The training script runs for a **fixed time budget of 120 seconds** (wall clock training time only — data loading before the loop and evaluation after it are not counted). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, masking rate, seed, device, thread count, how the three task losses are weighted and combined during training, whether the three heads share more or less of the trunk, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, parameter cap, etc).
- Change the fixed model interface contract: your model must expose `forward_mlm(input_ids, attention_mask, segment_ids, selected)`, `forward_lm(input_ids, attention_mask, segment_ids)`, and `forward_nsp(input_ids, attention_mask, segment_ids)` — `evaluate_multitask` in `prepare.py` calls exactly these three methods. You can change everything about what happens *inside* them.
- Change the fixed composite-score weights (`mlm_loss + lm_loss + 0.5 * nsp_loss`, defined in `evaluate_multitask`). You may freely retune the *training-time* loss weights in `train.py` (`MLM_LOSS_WEIGHT` / `LM_LOSS_WEIGHT` / `NSP_LOSS_WEIGHT`) — those only affect what the optimizer sees, not how a run is scored.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_multitask` function in `prepare.py` is the ground truth metric, and the `val_*.pt` files are the fixed comparison set — never regenerate them mid-campaign.
- Read held-out test examples or labels during the search, train on validation/test data, hardcode answers, or otherwise adapt model behavior to the evaluation set. Use only the aggregate `val_score` as feedback.

**The goal is simple: get the lowest `val_score`** (the fixed composite: `mlm_loss + lm_loss + 0.5 * nsp_loss`). Since the time budget is fixed, you don't need to worry about training time — it's always 120 seconds. The only hard constraints are that the code runs without crashing, finishes within the budget, stays under the 3M parameter cap enforced in `train.py`, and keeps the `forward_mlm` / `forward_lm` / `forward_nsp` contract intact.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_score:        7.845210
val_mlm_loss:     3.912345
val_mlm_accuracy: 0.298000
val_lm_loss:      3.801122
val_lm_accuracy:  0.312000
val_nsp_loss:     0.663480
val_nsp_accuracy: 0.611000
training_seconds: 120.1
total_seconds:    126.3
peak_rss_mb:      640.2
total_tokens_M:   6.144
selected_tokens_M: 0.921
num_steps:        410
num_params_M:     1.1
depth:            2
```

`val_score` is the single number that matters for keep/discard decisions — it's the fixed composite defined in `prepare.py`. The other `val_*` lines are diagnostic: useful for understanding *why* a change helped or hurt (e.g. an idea that improves MLM but tanks NSP), but not themselves the target.

Note that the script always stops after 120 seconds of training, so depending on the computing device the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_score:" run.log
```

If the grep output is empty, the run crashed — see "Crashes" below.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_score	memory_mb	status	description
```

1. git commit hash (short, 7 chars)
2. val_score achieved (e.g. 7.845210) — use 0.000000 for crashes
3. peak host memory in MB, round to .1f (from `peak_rss_mb`) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_score	memory_mb	status	description
a1b2c3d	8.100000	600.0	keep	baseline
b2c3d4e	7.950000	605.2	keep	increase LR to 1e-3
c3d4e5f	8.200000	600.0	discard	switch to ReLU activation
d4e5f6g	0.000000	0.0	crash	double hidden size (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit.
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
5. Read out the results: `grep "^val_score:\|^peak_rss_mb:" run.log`.
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
7. Record the results in the tsv (NOTE: do not commit results.tsv, leave it untracked by git).
8. If val_score improved (lower), you "advance" the branch, keeping the git commit.
9. If val_score is equal or worse, restore `train.py` from the previous accepted commit (`git checkout <prev-commit> -- train.py`) and commit that restoration. Never use a destructive repository-wide reset.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~120 seconds total (+ a few seconds for startup and eval overhead — eval now runs three passes over the val set, so overhead is a bit higher than a single-task setup). If a run seems stuck well past that, interrupt it, treat it as a failure, and move on.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**Confirming a candidate**: A lower single-seed score is a provisional improvement, not proof. Before treating a candidate as the new accepted baseline for a *major* architectural or optimizer change, rerun it once or twice more with `SEED` changed to 29 and 43 directly in `train.py` (revert the seed afterward) and check the improvement holds on average. Minor tweaks (small LR nudges, loss-weight nudges, etc.) don't need this — use judgment.

**Held-out test set**: `test_clean.pt` / `test_mlm_inputs.pt` / `test_mlm_targets.pt` exist in `.data/bert/` but must not be touched during the search. Only if the user explicitly authorizes a final check on the strongest candidate, temporarily point the evaluation load in `train.py` at the `test_*` files instead of the `val_*` files, run once, record the result, then revert the change. Never tune further based on that number.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes (including how the three tasks interact: how much of the trunk they share, how their gradients are balanced, whether one task's representations should feed another). The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each experiment now costs three forward/backward passes per step instead of one, so expect noticeably fewer optimizer steps per 120s run than a single-task setup — that's expected and not itself a bug. If each experiment takes ~2-3 minutes then you can run roughly 20-25/hour. The user then wakes up to experimental results, all completed by you while they slept!
