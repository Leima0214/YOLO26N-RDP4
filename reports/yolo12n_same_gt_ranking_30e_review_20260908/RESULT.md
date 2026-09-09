# YOLO12n TAL-positive same-GT ranking 30E review

## Verdict

**NO-GO.** The training-only pairwise ranking loss did not improve the matched
30E detector and did not improve its intended validation-set ordering
mechanism. Do not promote this implementation and do not start a weight or
margin rescue scan.

## Matched protocol

- SVRDD7 Train 6000 / Val 1000; Test was not read.
- Official YOLO12n, native non-end2end Detect, TAL, BCE, CIoU and DFL.
- 30 epochs, 640 px, batch 32, seed 42, optimizer=auto/MuSGD, native augmentation.
- Checkpoint selection: native validation best.pt; both runs selected epoch 30.
- Final comparison: canonical COCO Val, score floor 0.001, NMS IoU 0.7,
  max_det 300, COCO maxDets 100, FP32, no Voting.
- Only treatment: same-GT ranking on native TAL positives with IoU gap >=0.05,
  loss weight 0.10 and no new model parameters.

## Canonical no-Voting result

All values are percentages; deltas are percentage points.

| Metric | YOLO12n B0 30E | Same-GT Rank 30E | Delta |
|---|---:|---:|---:|
| AP50:95 | 38.338 | 37.891 | **-0.447** |
| AP50 | 63.428 | 63.382 | -0.046 |
| AP75 | 40.456 | 39.918 | **-0.538** |
| AP-small | 14.800 | 13.918 | **-0.881** |
| AP-medium | 32.833 | 33.174 | +0.341 |
| AP-large | 53.645 | 58.694 | +5.049 |
| AR100 | 59.864 | 59.737 | -0.127 |

AP-large increased, but the route fails the main gate: overall AP, AP75,
AP-small and AR100 all fell. Large-scale category supports are uneven; the
large gain cannot override the matched aggregate regression.

## Per-class deltas

| Class | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| LC | -0.231 | +0.474 | -0.478 | +0.255 |
| TC | **-1.777** | **-2.451** | **-2.248** | -0.718 |
| AC | **-1.064** | -0.299 | **-3.236** | +0.676 |
| P | -0.259 | +2.352 | -0.624 | -1.478 |
| MC | -0.008 | -1.859 | +2.499 | +0.347 |
| LP | +0.118 | +0.396 | +0.664 | +0.058 |
| TP | +0.092 | +1.061 | -0.345 | -0.029 |

Only LP and TP gained total AP, while the largest regressions occurred on TC
and AC. The intervention did not produce a coherent class-wide improvement.

## Matched mechanism check

The old 100E diagnostic was not used as the causal control. The official
YOLO12n 30E best.pt was rerun through the same TAL-positive audit.

| Validation mechanism | B0 30E | Rank 30E | Delta |
|---|---:|---:|---:|
| Mean best positive IoU | 0.82105 | 0.81886 | -0.00218 |
| Mean top-score positive IoU | 0.72397 | 0.72373 | -0.00023 |
| Best minus top-score IoU | 0.09708 | 0.09513 | -0.00195 |
| Score-IoU within-GT Spearman | -0.04729 | -0.05795 | -0.01066 |
| Score weighted pair accuracy | 0.42928 | 0.41797 | **-0.01130** |

The top-score quality was effectively unchanged and pairwise ordering became
worse. The tiny reduction in the best/top gap came from slightly lower best
positive IoU rather than a better top-scored box.

By scale, pair accuracy changed by -0.0014 for small, -0.0102 for medium and
-0.0180 for large. Small best-positive IoU also fell from 0.67064 to 0.66509.

## Cost and artifacts

- Parameters: 2,569,413 in both runs; inference graph and FLOPs unchanged.
- B0 training time: 2084.19 s (34.74 min).
- Ranking training time: 2537.07 s (42.28 min).
- Added training time: 452.88 s (+21.7%).
- Candidate best.pt SHA256:
  `ea91635adbbc0f008051f9de86bd25ff5079e9e5aeb219ea5605748623d45e39`.
- Candidate last.pt SHA256:
  `dff29a52a480755dee1694a57b772d7ce467e75aec2b8e214e9c25eec81a4bd5`.
- Training exit code 0; canonical evaluation exit code 0; matched mechanism
  diagnostic exit code 0.

## Failure analysis and next route

The original diagnosis was valid: native scores poorly order positives within
one GT. This experiment shows that directly imposing a pairwise IoU order is
not an effective repair.

The auxiliary target is endogenous because it is rebuilt from the detector's
current predicted boxes every iteration. It therefore moves with training.
It also competes with native TAL/BCE targets, which already combine class
confidence and a strongly IoU-weighted alignment metric. Plain-IoU pairs remove
that semantic component. Hundreds of thousands of redundant pairs per epoch
then send extra classification gradients through shared neck features. This
can disturb candidate geometry even though the added loss acts on class logits.

The next priority should move to candidate formation for small damage. The
matched B0 has small best-positive IoU 0.67064 versus 0.83164 for medium and
0.88751 for large, and the ranking treatment reduced small AP further. First
perform a YOLO12-specific P2-to-P3 early-detail survival audit. If P2 has useful
small/thin-target contrast that collapses at P3, develop one bounded,
semantic-gated P2-to-P3 residual detail injection. If P3 retains the signal,
return to a separate scale-conditioned DFL quality-calibration probe. Do not
combine either route with this rejected ranking loss.

