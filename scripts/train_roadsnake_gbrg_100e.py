"""Fresh-start Japan4-cleanV3 100E RoadSnake-GBRG formal experiment."""

from __future__ import annotations

import train_roadsnake_gbrg_30e as experiment

experiment.EPOCHS = 100
experiment.RUN_NAME = "yolo26n-japan4-roadsnake-gbrg_cleanv3_100e_seed42_20260823"


if __name__ == "__main__":
    experiment.main()
