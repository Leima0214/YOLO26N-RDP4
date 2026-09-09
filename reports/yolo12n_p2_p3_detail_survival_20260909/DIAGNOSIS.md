# YOLO12n P2-to-P3 early-detail survival diagnosis

Decision: **NO_GO_DETAIL_INJECTION_GO_P3_LOCALIZATION_AUDIT**

This is a training-free SVRDD7 Val audit. Test was not read and no visualization was generated.

## Discovered stages

| Stage | Layer | Module | Stride | Shape | Channels |
|---|---:|---|---:|---|---:|
| P2 | 2 | C3k2 | 4 | [1, 64, 168, 168] | 64 |
| P3 | 4 | C3k2 | 8 | [1, 128, 84, 84] | 128 |
| P4 | 6 | A2C2f | 16 | [1, 128, 42, 42] | 128 |

## Small-object evidence

| Group | Stage | n GT | Contrast mean | 95% CI | Probe AUC | Paired accuracy |
|---|---|---:|---:|---|---:|---:|
| small | P2 | 548 | 0.08217 | [0.07219, 0.09199] | 0.8885 | 0.9197 |
| small | P3 | 548 | 0.20644 | [0.18995, 0.22300] | 0.9152 | 0.9580 |
| small | P4 | 548 | 0.20948 | [0.19279, 0.22640] | 0.7876 | 0.9270 |
| small_hit | P2 | 440 | 0.08994 | [0.07897, 0.10103] | 0.9073 | 0.9432 |
| small_hit | P3 | 440 | 0.23549 | [0.21654, 0.25389] | 0.9415 | 0.9705 |
| small_hit | P4 | 440 | 0.23468 | [0.21587, 0.25453] | 0.8242 | 0.9295 |
| small_nonhit | P2 | 108 | 0.05049 | [0.02967, 0.07139] | 0.8022 | 0.8519 |
| small_nonhit | P3 | 108 | 0.08806 | [0.06424, 0.11287] | 0.7951 | 0.9352 |
| small_nonhit | P4 | 108 | 0.10682 | [0.08243, 0.13322] | 0.6522 | 0.8889 |
| small_localization_error | P2 | 104 | 0.05173 | [0.03052, 0.07378] | 0.7949 | 0.8558 |
| small_localization_error | P3 | 104 | 0.08955 | [0.06518, 0.11476] | 0.7925 | 0.9423 |
| small_localization_error | P4 | 104 | 0.10877 | [0.08337, 0.13758] | 0.6515 | 0.8846 |
| small_complete_miss | P2 | 1 | 0.08307 | [0.08307, 0.08307] | NA | NA |
| small_complete_miss | P3 | 1 | 0.03052 | [0.03052, 0.03052] | NA | NA |
| small_complete_miss | P4 | 1 | 0.04109 | [0.04109, 0.04109] | NA | NA |
| elongated_ar_ge_5 | P2 | 695 | -0.04390 | [-0.05298, -0.03536] | 0.8850 | 0.9036 |
| elongated_ar_ge_5 | P3 | 695 | 0.03444 | [0.02716, 0.04158] | 0.9336 | 0.9640 |
| elongated_ar_ge_5 | P4 | 695 | 0.10237 | [0.09286, 0.11238] | 0.9385 | 0.9669 |

## P2-to-P3 paired change

| Group | n GT | Delta contrast | 95% CI |
|---|---:|---:|---|
| small | 548 | 0.12427 | [0.11022, 0.13817] |
| elongated_ar_ge_5 | 695 | 0.07834 | [0.07094, 0.08578] |
| small_hit | 440 | 0.14555 | [0.12961, 0.16137] |
| small_nonhit | 108 | 0.03757 | [0.01342, 0.06198] |
| small_localization_error | 104 | 0.03783 | [0.01341, 0.06289] |
| small_complete_miss | 1 | -0.05255 | [-0.05255, -0.05255] |

## Adjudication

- small_nonhit remains linearly separable at both P2 and P3 without the predeclared collapse pattern.
- Next action: Audit P3 regression, DFL distributions, and scale-conditioned localization before changing features.
- Current same-GT residual ranking remains rejected and is not part of this route.

## Limits

Feature contrast and a frozen linear probe are representation proxies, not causal proof that a trainable injection will improve AP. Stage receptive fields differ. Any proposed module still requires a fresh matched training comparison.
