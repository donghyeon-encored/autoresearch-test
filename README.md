# Results (sep1)

| commit | val_score | memory_mb | status | description |
|---|---|---|---|---|
| f598309 | 11.487637 | 825.9 | keep | baseline |
| 160e8e4 | 11.251428 | 1099.9 | keep | fuse NSP into MLM forward pass (share bidirectional encode) |
| 4c56b34 | 11.496332 | 973.5 | discard | bf16 autocast for training forward (MPS overhead outweighs benefit, fewer steps) |
| 2037969 | 11.113171 | 808.9 | keep | batch 16x4accum -> 64x1accum (fewer kernel launches, same effective batch) |
| d4a920e | 10.966271 | 1462.9 | keep | device batch 64->32, accum=1 (more/noisier steps helped: 1462 steps) |
| cdb2747 | 10.895472 | 1410.3 | keep | device batch 32->16, accum=1 (2353 steps) |
| 8d36363 | 11.112660 | 1039.6 | discard | device batch 16->8, accum=1 (regressed, too noisy/small) |
| 7c5dd51 | 11.040253 | 1008.3 | discard | LR 2e-3->1.5e-3 for batch16 regime (regressed) |
| f12083d | 10.926663 | 1100.0 | discard | warmup 0.1->0.05 for batch16 regime (regressed slightly) |
| addfdc6 | 10.992715 | 943.9 | discard | depth 2->3 (fewer steps 1931 outweighed added capacity) |
| 7d5a0f0 | 10.917692 | 970.9 | discard | intermediate_size 512->256 (steps barely increased, capacity loss dominated) |
| 6994888 | 10.917987 | 1001.0 | discard | merge LM+MLM/NSP into one combined-batch(32) encode call (regressed, fewer/larger-batch steps) |
| 90978d2 | 11.099796 | 1050.0 | discard | hidden_size 128->96 (steps dropped too, capacity loss dominated) |
| 41f2a0b | 10.952486 | 1079.0 | discard | MLM_LOSS_WEIGHT 1.0->1.5 (regressed slightly) |
| 3e32c08 | 10.867631 | 911.8 | keep | mask_rate 0.6->0.5 for batch16 regime (improved) |
| f2a8506 | 10.830801 | 904.7 | keep | mask_rate 0.5->0.4 (improved again) |
| 5a05a68 | 10.786513 | 935.6 | keep | mask_rate 0.4->0.3 (improved again) |
| 26d2bcc | 10.924679 | 938.9 | discard | mask_rate 0.3->0.2 (regressed, 0.3 is the local optimum) |
| 4c9f670 | 10.859328 | 814.0 | discard | mask_rate 0.3->0.25 (regressed, 0.3 confirmed local optimum) |
| 517a331 | 9.380421 | 919.0 | keep | dropout 0.1->0.0 (big win, model underfit not overfit at this budget) |
| 76175b4 | 10.442702 | 953.3 | discard | depth 2->3 retest with dropout=0.0 (still regresses, fewer steps dominates) |
| b5f88db | 9.321763 | 955.2 | keep | LR 2e-3->2.5e-3 (retest with dropout=0.0, slight improvement) |
| dbbf947 | 9.159252 | 847.0 | keep | LR 2.5e-3->3e-3 (further improvement) |

Best so far: `dbbf947` — val_score 9.159252.
