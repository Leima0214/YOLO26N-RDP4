"""Fresh-start 100E GBRG-A: unchanged GBRG through E50, cosine decay to zero at E100."""

from __future__ import annotations

import train_roadsnake_gbrg_30e as experiment

experiment.EPOCHS = 100
experiment.GBRG_ANNEAL_START_EPOCH = 50
experiment.GBRG_ANNEAL_END_EPOCH = 100
experiment.RUN_NAME = "yolo26n-japan4-roadsnake-gbrg-a_cleanv3_100e_seed42_20260823"


if __name__ == "__main__":
    experiment.main()
