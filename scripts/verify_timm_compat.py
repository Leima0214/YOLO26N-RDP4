"""Smoke-check the timm APIs used by VanillaNet without importing all optional YOLO modules."""

import runpy
from pathlib import Path

import timm


module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "ultralytics/nn/vanillanet.py"))
assert callable(module["trunc_normal_"])
assert module["DropPath"] is not None
assert callable(module["register_model"])
print(f"timm {timm.__version__} compatibility: OK")
