#!/usr/bin/env python3
"""Train the frozen 100E SVRDD7 YOLO26n baseline; selection is Val-only."""

from svrdd7_training_common import run_training

if __name__ == "__main__":
    run_training("baseline")
