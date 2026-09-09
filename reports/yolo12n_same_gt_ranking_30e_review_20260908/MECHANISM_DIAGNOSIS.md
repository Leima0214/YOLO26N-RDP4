# YOLO12n native TAL-positive and DFL uncertainty diagnosis

## Protocol

- Frozen checkpoint: `/root/YOLO26N-RDP4/runs/paper1_svrdd7_yolo12_ranking/YOLO12N_TAL_SAMEGT_RANK_30E_seed42/weights/best.pt`
- SHA256: `ea91635adbbc0f008051f9de86bd25ff5079e9e5aeb219ea5605748623d45e39`
- SVRDD7 Val only: 1000 images
- Model path: official non-end2end `Detect`, native TAL topk=10, alpha=0.5, beta=6.0, DFL reg_max=16
- Image size: 640; batch: 32; no Voting
- Test accessed: **false**; detector training performed: **false**
- Native decode reconstruction max absolute difference: `0.00024414` pixels

## Coverage

- GT with at least one native TAL positive: **2611**
- Native TAL positive candidates: **23794**
- Mean positives per represented GT: **9.113**

## Candidate-level quality

| Scope | Pos | GT | Mean IoU | Score rho | TAL target rho | Peak rho | Margin rho | -Entropy rho | -Variance rho |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 23794 | 2611 | 0.7445 | 0.5298 | 0.7133 | 0.3721 | 0.2700 | 0.4281 | 0.4513 |
| P3 | 9717 | 1593 | 0.6779 | 0.5145 | 0.7411 | 0.2798 | 0.1681 | 0.3319 | 0.3879 |
| P4 | 8962 | 1839 | 0.7705 | 0.4964 | 0.6763 | 0.3518 | 0.2467 | 0.4306 | 0.4662 |
| P5 | 5115 | 824 | 0.8253 | 0.3603 | 0.5980 | 0.3569 | 0.2902 | 0.4499 | 0.4503 |

| Size | Pos | GT | Mean IoU | IoU>=.75 | Score rho | Peak rho | -Entropy rho |
|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 3332 | 547 | 0.5174 | 0.2188 | 0.6357 | 0.4045 | 0.5381 |
| medium | 9636 | 980 | 0.7371 | 0.5641 | 0.4825 | 0.4767 | 0.5448 |
| large | 10826 | 1084 | 0.8209 | 0.7597 | 0.4125 | 0.4302 | 0.5347 |

`rho` is Spearman correlation with the candidate's plain IoU to its assigned GT. Entropy and variance are negated in the certainty columns, so a positive value has the expected direction.

## Within-GT ordering

| GT | Pos/GT | Best IoU | Top-score IoU | Gap | Score pair acc | TAL target pair acc | Peak pair acc | -Entropy pair acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2611 | 9.1130 | 0.8189 | 0.7237 | 0.0951 | 0.4180 | 0.8428 | 0.4934 | 0.4877 |

Pairwise accuracy uses only positive pairs whose IoU differs by at least 0.05. A value of 0.5 is chance ordering.

## Incremental DFL evidence

| DFL certainty | Pooled rho | Partial r | Within-GT pair acc | Levels rho>=.10 |
|---:|---:|---:|---:|---:|
| dfl_peak_mean | 0.3721 | 0.3776 | 0.4934 | 3 |
| dfl_margin_mean | 0.2700 | 0.2347 | 0.4959 | 3 |
| dfl_neg_entropy_mean | 0.4281 | 0.4903 | 0.4877 | 3 |
| dfl_neg_variance_mean | 0.4513 | 0.3767 | 0.4891 | 3 |

`Partial r` controls native class score, P3/P4/P5 level, and log GT area. This is the key check against a misleading pooled scale correlation.

## Monotonicity by IoU bin

| IoU bin | N | Peak | Margin | Entropy | Variance | Pred score | TAL target |
|---:|---:|---:|---:|---:|---:|---:|---:|
| [0.0,0.5) | 2688 | 0.5995 | 0.3419 | 0.3528 | 0.0055 | 0.0510 | 0.0762 |
| [0.5,0.6) | 1743 | 0.6044 | 0.3394 | 0.3413 | 0.0046 | 0.1328 | 0.2385 |
| [0.6,0.7) | 2966 | 0.6132 | 0.3448 | 0.3312 | 0.0038 | 0.1672 | 0.3402 |
| [0.7,0.8) | 4551 | 0.6337 | 0.3614 | 0.3072 | 0.0029 | 0.2281 | 0.4608 |
| [0.8,0.9) | 7000 | 0.6678 | 0.3957 | 0.2728 | 0.0018 | 0.3569 | 0.6060 |
| [0.9,1.0] | 4846 | 0.7042 | 0.4441 | 0.2409 | 0.0012 | 0.4937 | 0.7318 |

## Decisions

### DFL-guided quality estimation: **REVIEW**

- Best diagnostic feature: `dfl_neg_entropy_mean`
- Checks: `{"pooled_spearman_ge_015": true, "partial_pearson_ge_005": true, "within_gt_pairwise_ge_053": false, "two_levels_ge_010": true}`
- Interpretation: a GO requires pooled, conditional, within-GT, and cross-level evidence together. REVIEW means some useful signal exists but a DFL-only quality branch is not yet justified.

### Same-GT residual ranking supervision: **GO**

- Native score pairwise accuracy: `0.4180`
- Native TAL target pairwise accuracy: `0.8428`
- Checks: `{"native_pairwise_below_070": true, "target_exceeds_native_by_005": true, "mean_quality_gap_ge_003": true}`
- Interpretation: the TAL target already encodes quality. A GO means the learned score fails to realize a materially stronger target ordering, leaving a specific residual-ranking hypothesis.

## Limits

- This audit evaluates actual positives selected by the frozen model's native TAL. It does not claim that a new quality module will recover the GT-IoU oracle.
- Correlation is mechanism evidence, not a performance result.
- DFL statistics may identify localization certainty but cannot by themselves prove foreground semantics or suppress every background false positive.
- One checkpoint and one Val split authorize or reject development; they do not establish a final paper claim.
