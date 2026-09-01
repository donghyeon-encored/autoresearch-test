# Results (easy)

| commit | val_mlm_loss | memory_mb | status | description |
|---|---|---|---|---|
| aa7afb2 | 6.267576 | 724.2 | keep | baseline |
| 950973e | 6.083134 | 673.7 | keep | LR 5e-4 -> 1e-3 |
| dfc1148 | 5.999044 | 919.9 | keep | LR 1e-3 -> 2e-3 |
| 954285d | 6.025272 | 723.7 | discard | LR 2e-3 -> 3e-3 |
| 695147b | 6.038838 | 764.4 | discard | warmup ratio 0.1 -> 0.03 |
| 847521a | 6.003459 | 710.2 | discard | disable dropout |
| 3fb1909 | 5.999224 | 810.4 | discard | grad accum 4 -> 2 (eff batch 32) |
| 577c755 | 6.065282 | 616.2 | discard | hidden 128->192, intermediate 512->768 |
| 10c80df | 6.028578 | 794.4 | discard | depth 2 -> 1 |
| 5a14306 | 6.001261 | 757.7 | discard | adam beta2 0.999 -> 0.98 |
| cccd5a8 | 6.013557 | 669.8 | discard | cosine LR decay instead of linear |
| 8a71cc4 | 6.002252 | 738.6 | discard | LR 2e-3 -> 2.5e-3 |
| 3b4d890 | 6.007407 | 766.3 | discard | num_heads 4 -> 8 |
| 88419af | 5.998675 | 704.3 | keep | weight decay 0.01 -> 0.0 |
| 006ef02 | 5.980236 | 755.5 | keep | mask rate 0.15 -> 0.2 |
| 13915d0 | 5.965307 | 848.8 | keep | mask rate 0.2 -> 0.3 |
| cc65b60 | 5.957382 | 1095.0 | keep | mask rate 0.3 -> 0.4 |
| fcd47f0 | 5.951346 | 878.9 | keep | mask rate 0.4 -> 0.5 |
| 25f3886 | 5.947340 | 880.8 | keep | mask rate 0.5 -> 0.6 |
| 6e0731e | 5.957618 | 1118.0 | discard | mask rate 0.6 -> 0.75 |
| 6d6c86d | 5.952485 | 1221.0 | discard | mask rate 0.6 -> 0.7 |
| 5e96fb3 | 5.959502 | 938.3 | discard | warmup 0.1 -> 0.05 (retune with mask_rate=0.6) |

Best so far: `25f3886` — val_mlm_loss 5.947340.
