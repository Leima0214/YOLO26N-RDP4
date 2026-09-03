#!/usr/bin/env python3
"""Train the frozen 100E SVRDD7 RoadSnake-R1 control; selection is Val-only."""

from svrdd7_training_common import run_training

if __name__ == "__main__":
    run_training("roadsnake_r1")
