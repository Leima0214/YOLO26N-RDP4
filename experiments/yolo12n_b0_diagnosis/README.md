# YOLO12n B0 diagnosis

This directory contains a Val-only, training-free diagnosis of the frozen official YOLO12n 100E checkpoint. Test was not accessed.

1. `scripts/diagnose_yolo12n_b0_route.py` exports raw/final candidate, error, scale, aspect-ratio, FPN and gallery evidence.
2. `scripts/analyze_yolo12n_b0_oracle_ranking.py` verifies stored-prediction parity and computes the GT-IoU score oracle upper bound.
3. `diagnostic_report.md` and `route_decision.md` contain the final post-oracle adjudication.

The oracle is an unavailable upper bound and is never reported as expected model gain.
