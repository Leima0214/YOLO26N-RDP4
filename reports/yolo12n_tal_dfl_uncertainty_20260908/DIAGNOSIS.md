# YOLO12n native TAL-positive and DFL uncertainty diagnosis

## Protocol

- Frozen checkpoint: `/root/YOLO26N-RDP4/runs/paper1_svrdd7_official/YOLO12N_OFFICIAL_SVRDD7_100E_seed42/weights/best.pt`
- SHA256: `76a35d4bd5b7c9ed499c30b9b272d2c206ea10f81273ce4b5ccd4b040d38690b`
- SVRDD7 Val only: 1000 images
- Model path: official non-end2end `Detect`, native TAL topk=10, alpha=0.5, beta=6.0, DFL reg_max=16
- Image size: 640; batch: 32; no Voting
- Test accessed: **false**; detector training performed: **false**
- Native decode reconstruction max absolute difference: `0.00024414` pixels

## Coverage

- GT with at least one native TAL positive: **2611**
- Native TAL positive candidates: **23818**
- Mean positives per represented GT: **9.122**

## Candidate-level quality

| Scope | Pos | GT | Mean IoU | Score rho | TAL target rho | Peak rho | Margin rho | -Entropy rho | -Variance rho |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 23818 | 2611 | 0.7437 | 0.5230 | 0.7144 | 0.3635 | 0.2669 | 0.4408 | 0.4626 |
| P3 | 9512 | 1561 | 0.6851 | 0.5128 | 0.7325 | 0.2495 | 0.1414 | 0.3153 | 0.3700 |
| P4 | 9054 | 1847 | 0.7606 | 0.4999 | 0.6943 | 0.3588 | 0.2565 | 0.4621 | 0.4856 |
| P5 | 5252 | 795 | 0.8206 | 0.3539 | 0.6230 | 0.3789 | 0.3171 | 0.4786 | 0.4829 |

| Size | Pos | GT | Mean IoU | IoU>=.75 | Score rho | Peak rho | -Entropy rho |
|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 3371 | 547 | 0.5267 | 0.2332 | 0.6712 | 0.3746 | 0.5045 |
| medium | 9633 | 980 | 0.7400 | 0.5584 | 0.4726 | 0.4413 | 0.5335 |
| large | 10814 | 1084 | 0.8146 | 0.7382 | 0.4018 | 0.4299 | 0.5338 |

`rho` is Spearman correlation with the candidate's plain IoU to its assigned GT. Entropy and variance are negated in the certainty columns, so a positive value has the expected direction.

## Within-GT ordering

| GT | Pos/GT | Best IoU | Top-score IoU | Gap | Score pair acc | TAL target pair acc | Peak pair acc | -Entropy pair acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2611 | 9.1222 | 0.8213 | 0.7302 | 0.0911 | 0.4313 | 0.8388 | 0.4928 | 0.4867 |

Pairwise accuracy uses only positive pairs whose IoU differs by at least 0.05. A value of 0.5 is chance ordering.

## Incremental DFL evidence

| DFL certainty | Pooled rho | Partial r | Within-GT pair acc | Levels rho>=.10 |
|---:|---:|---:|---:|---:|
| dfl_peak_mean | 0.3635 | 0.3189 | 0.4928 | 3 |
| dfl_margin_mean | 0.2669 | 0.1997 | 0.4986 | 3 |
| dfl_neg_entropy_mean | 0.4408 | 0.4269 | 0.4867 | 3 |
| dfl_neg_variance_mean | 0.4626 | 0.3046 | 0.4907 | 3 |

`Partial r` controls native class score, P3/P4/P5 level, and log GT area. This is the key check against a misleading pooled scale correlation.

## Monotonicity by IoU bin

| IoU bin | N | Peak | Margin | Entropy | Variance | Pred score | TAL target |
|---:|---:|---:|---:|---:|---:|---:|---:|
| [0.0,0.5) | 2710 | 0.6142 | 0.3550 | 0.3348 | 0.0043 | 0.0681 | 0.0804 |
| [0.5,0.6) | 1902 | 0.6148 | 0.3458 | 0.3273 | 0.0038 | 0.1906 | 0.2670 |
| [0.6,0.7) | 2970 | 0.6216 | 0.3505 | 0.3194 | 0.0035 | 0.2276 | 0.3675 |
| [0.7,0.8) | 4572 | 0.6468 | 0.3752 | 0.2926 | 0.0025 | 0.2951 | 0.4823 |
| [0.8,0.9) | 6627 | 0.6772 | 0.4071 | 0.2616 | 0.0017 | 0.4223 | 0.6211 |
| [0.9,1.0] | 5037 | 0.7139 | 0.4570 | 0.2292 | 0.0011 | 0.5742 | 0.7514 |

## Decisions

### DFL-guided quality estimation: **REVIEW**

- Best diagnostic feature: `dfl_neg_entropy_mean`
- Checks: `{"pooled_spearman_ge_015": true, "partial_pearson_ge_005": true, "within_gt_pairwise_ge_053": false, "two_levels_ge_010": true}`
- Interpretation: a GO requires pooled, conditional, within-GT, and cross-level evidence together. REVIEW means some useful signal exists but a DFL-only quality branch is not yet justified.

### Same-GT residual ranking supervision: **GO**

- Native score pairwise accuracy: `0.4313`
- Native TAL target pairwise accuracy: `0.8388`
- Checks: `{"native_pairwise_below_070": true, "target_exceeds_native_by_005": true, "mean_quality_gap_ge_003": true}`
- Interpretation: the TAL target already encodes quality. A GO means the learned score fails to realize a materially stronger target ordering, leaving a specific residual-ranking hypothesis.

## Limits

- This audit evaluates actual positives selected by the frozen model's native TAL. It does not claim that a new quality module will recover the GT-IoU oracle.
- Correlation is mechanism evidence, not a performance result.
- DFL statistics may identify localization certainty but cannot by themselves prove foreground semantics or suppress every background false positive.
- One checkpoint and one Val split authorize or reject development; they do not establish a final paper claim.
