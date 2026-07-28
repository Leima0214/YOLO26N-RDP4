"""Build a class-balanced Japan7 P3 dictionary from pretrained YOLO26n features."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402

BASELINE_YAML = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/japan7_remote.yaml")
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--output", default="reports/japan7_rd_p3_atoms.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--atoms", type=int, default=128)
    parser.add_argument("--samples-per-group", type=int, default=8192)
    parser.add_argument("--samples-per-box", type=int, default=32)
    parser.add_argument("--background-per-image", type=int, default=32)
    parser.add_argument("--kmeans-iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(value: str) -> torch.device:
    if value.lower() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required unless --device cpu is specified")
    index = int(value.split(",")[0])
    return torch.device(f"cuda:{index}")


def update_reservoir(
    vectors: torch.Tensor | None,
    keys: torch.Tensor | None,
    incoming: torch.Tensor,
    capacity: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep a uniform priority reservoir without per-sample Python loops."""
    incoming = incoming.detach().float().cpu()
    incoming_keys = torch.rand(incoming.shape[0], generator=generator)
    vectors = incoming if vectors is None else torch.cat((vectors, incoming), dim=0)
    keys = incoming_keys if keys is None else torch.cat((keys, incoming_keys), dim=0)
    if vectors.shape[0] > capacity:
        keep = keys.topk(capacity, sorted=False).indices
        vectors, keys = vectors[keep], keys[keep]
    return vectors, keys


def sample_rows(
    values: torch.Tensor, limit: int, generator: torch.Generator
) -> torch.Tensor:
    if values.shape[0] <= limit:
        return values
    return values[
        torch.randperm(values.shape[0], generator=generator, device="cpu")[:limit].to(
            values.device
        )
    ]


def focused_box(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    """Keep the long crack axis but trim background along its short axis."""
    width, height = x2 - x1, y2 - y1
    if width >= 2 * height:
        trim = height // 4
        return x1, y1 + trim, x2, max(y1 + trim + 1, y2 - trim)
    if height >= 2 * width:
        trim = width // 4
        return x1 + trim, y1, max(x1 + trim + 1, x2 - trim), y2
    trim_x, trim_y = width // 10, height // 10
    return (
        x1 + trim_x,
        y1 + trim_y,
        max(x1 + trim_x + 1, x2 - trim_x),
        max(y1 + trim_y + 1, y2 - trim_y),
    )


def collect_feature_samples(
    model,
    loader,
    device: torch.device,
    class_count: int,
    capacity: int,
    per_box: int,
    background_per_image: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[int]]:
    reservoirs: list[torch.Tensor | None] = [None] * (class_count + 1)
    priorities: list[torch.Tensor | None] = [None] * (class_count + 1)
    seen = [0] * (class_count + 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = model.model[4].register_forward_hook(hook)
    try:
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True).float() / 255
            captured.clear()
            with torch.inference_mode():
                model(images)
            if len(captured) != 1:
                raise RuntimeError(f"expected one P3 hook output, got {len(captured)}")
            features = captured[0]
            batch_idx = batch["batch_idx"].long()
            classes = batch["cls"].view(-1).long()
            boxes = batch["bboxes"].float()

            for image_index in range(features.shape[0]):
                grid = features[image_index].permute(1, 2, 0)
                height, width, channels = grid.shape
                image_targets = batch_idx == image_index
                image_boxes, image_classes = (
                    boxes[image_targets],
                    classes[image_targets],
                )
                background = torch.ones(
                    (height, width), dtype=torch.bool, device=features.device
                )

                for box, class_id_tensor in zip(image_boxes, image_classes):
                    class_id = int(class_id_tensor)
                    cx, cy, bw, bh = box.tolist()
                    x1 = max(0, min(width - 1, int((cx - bw / 2) * width)))
                    y1 = max(0, min(height - 1, int((cy - bh / 2) * height)))
                    x2 = max(x1 + 1, min(width, int((cx + bw / 2) * width + 0.999)))
                    y2 = max(y1 + 1, min(height, int((cy + bh / 2) * height + 0.999)))
                    background[y1:y2, x1:x2] = False
                    fx1, fy1, fx2, fy2 = focused_box(x1, y1, x2, y2)
                    region = grid[fy1:fy2, fx1:fx2].reshape(-1, channels)
                    sampled = sample_rows(region, per_box, generator)
                    seen[class_id] += sampled.shape[0]
                    reservoirs[class_id], priorities[class_id] = update_reservoir(
                        reservoirs[class_id],
                        priorities[class_id],
                        sampled,
                        capacity,
                        generator,
                    )

                background_values = grid[background]
                sampled_background = sample_rows(
                    background_values, background_per_image, generator
                )
                background_id = class_count
                seen[background_id] += sampled_background.shape[0]
                reservoirs[background_id], priorities[background_id] = update_reservoir(
                    reservoirs[background_id],
                    priorities[background_id],
                    sampled_background,
                    capacity,
                    generator,
                )
    finally:
        handle.remove()

    missing = [index for index, values in enumerate(reservoirs) if values is None]
    if missing:
        raise RuntimeError(f"no P3 samples collected for groups {missing}")
    return [values for values in reservoirs if values is not None], seen


def cosine_kmeans(
    samples: torch.Tensor,
    clusters: int,
    iterations: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    if samples.shape[0] < clusters:
        raise ValueError(f"need at least {clusters} samples, got {samples.shape[0]}")
    points = F.normalize(samples.to(device), dim=1, eps=1e-6)
    generator = torch.Generator(device=device).manual_seed(seed)
    first = torch.randint(points.shape[0], (1,), generator=generator, device=device)
    centers = [points[first]]
    for _ in range(1, clusters):
        similarity = points @ torch.cat(centers, dim=0).t()
        distance = (1 - similarity.max(dim=1).values).clamp_min(1e-6)
        next_index = torch.multinomial(distance, 1, generator=generator)
        centers.append(points[next_index])
    centers_tensor = torch.cat(centers, dim=0)

    for _ in range(iterations):
        assignment = (points @ centers_tensor.t()).argmax(dim=1)
        updated = []
        for cluster in range(clusters):
            members = points[assignment == cluster]
            updated.append(
                centers_tensor[cluster] if members.numel() == 0 else members.mean(dim=0)
            )
        next_centers = F.normalize(torch.stack(updated), dim=1, eps=1e-6)
        if torch.allclose(next_centers, centers_tensor, atol=1e-5, rtol=0):
            centers_tensor = next_centers
            break
        centers_tensor = next_centers
    return centers_tensor.cpu()


def main() -> None:
    args = parse_args()
    data_path, weights, output = (
        resolve(args.data),
        resolve(args.weights),
        resolve(args.output),
    )
    if not BASELINE_YAML.is_file() or not data_path.is_file() or not weights.is_file():
        raise FileNotFoundError((BASELINE_YAML, data_path, weights))
    if (
        args.imgsz <= 0
        or args.imgsz % 32
        or min(args.batch, args.samples_per_group, args.samples_per_box) <= 0
    ):
        raise ValueError(
            "imgsz must be a positive multiple of 32 and sampling arguments must be positive"
        )

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    data = check_det_dataset(str(data_path), autodownload=False)
    class_count = int(data["nc"])
    groups = class_count + 1
    if args.atoms % groups:
        raise ValueError(
            f"--atoms must be divisible by {groups} Japan7/background groups, got {args.atoms}"
        )

    cfg = get_cfg(
        overrides={
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "task": "detect",
            "rect": False,
            "cache": False,
            "classes": None,
            "fraction": 1.0,
            "single_cls": False,
        }
    )
    dataset = build_yolo_dataset(
        cfg, data["train"], args.batch, data, mode="val", rect=False, stride=32
    )
    loader = build_dataloader(dataset, args.batch, args.workers, shuffle=False, rank=-1)
    model = YOLO(str(BASELINE_YAML), task="detect")
    model.load(str(weights))
    network = model.model.to(device).eval()

    samples, seen = collect_feature_samples(
        network,
        loader,
        device,
        class_count,
        args.samples_per_group,
        args.samples_per_box,
        args.background_per_image,
        args.seed,
    )
    channels = samples[0].shape[1]
    atoms_per_group = args.atoms // groups
    centroids = torch.cat(
        [
            cosine_kmeans(
                group, atoms_per_group, args.kmeans_iters, device, args.seed + index
            )
            for index, group in enumerate(samples)
        ],
        dim=0,
    )
    if tuple(centroids.shape) != (args.atoms, channels):
        raise AssertionError(f"unexpected centroid shape {tuple(centroids.shape)}")

    names = [str(data["names"][index]) for index in range(class_count)] + ["background"]
    payload = {
        "centroids": centroids,
        "meta": {
            "data": str(data_path),
            "train": str(data["train"]),
            "weights": str(weights),
            "weights_sha256": sha256(weights),
            "layer": 4,
            "stride": 8,
            "imgsz": args.imgsz,
            "atoms": args.atoms,
            "channels": channels,
            "groups": names,
            "atoms_per_group": atoms_per_group,
            "sampling": "class-balanced_bbox_axis_focus_plus_background",
            "reservoir_samples": [int(group.shape[0]) for group in samples],
            "seen_samples": seen,
            "seed": args.seed,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        json.dumps(
            {"output": str(output), "output_sha256": sha256(output), **payload["meta"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
