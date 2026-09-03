# SVRDD7 external validation

This directory is the canonical ledger for the seven-class SVRDD experiment (`LC, TC, AC, P, MC, LP, TP`). It validates transfer of the Japan4-derived model family on a separate dataset; it is not a zero-shot claim.

The sequence is fixed: audit data -> create missing COCO annotations -> train matched 100E candidates -> select on Val -> optionally run native R10 and T1 pruning -> evaluate Test once after the decision is frozen. Do not compare absolute AP values across Japan4 and SVRDD.

Core RoadSnake and GBRG implementations are intentionally unchanged. The existing model YAMLs are reused, with their `nc: 80` serving only as a pretrained placeholder; every training entry asserts that the reconstructed head has `nc == 7`.
