"""Fine-tune a small detector on the reviewed sessions, and export it for the board.

    uv sync --extra train
    uv run train                      # yolo11n at 320, on datasets/yolo
    uv run train --export             # and write the ONNX the RKNN conversion takes

Choices worth knowing, all of them driven by what the first session's frames look like:

* **320×320 input.** The frames are 720×1280 portrait, and a duck at 2 m is ~40 px tall in a 320
  letterbox — small but findable, and this has to leave a 50 Hz control loop alone on a board whose
  NPU is under a TOPS. Bigger is the first dial to turn if recall at distance is the problem.
* **Rotation augmentation, and not much translation.** The camera rolls with the gait; it does not
  get moved around a scene. ±12° of roll is the real variation.
* **No vertical flip, no mosaic past the early epochs.** Ducks are always the right way up in a
  picture the daemon already rotated, and mosaic invents contact between objects that never touch.
* **The blur is in the data, not the augmentation.** Half of every session is motion blur, which is
  a better teacher than anything a blur transform would invent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path("datasets/yolo/data.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune and export the duck detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--model", default="yolo11n.pt", help="COCO-pretrained starting point")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--name", default="duck", help="run name under runs/")
    parser.add_argument("--device", default=None, help="cuda index, or cpu")
    parser.add_argument("--export", action="store_true", help="write ONNX when training finishes")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"no {args.data} — run `uv run dataset build` first")
    header = args.data.read_text().splitlines()[0]
    if "SMOKE" in header:
        print("!! smoke dataset: one session split by frame. The val numbers mean nothing.\n")

    # Imported here rather than at module scope: `--help` should work without the train extra
    # installed, and ultralytics takes seconds to import.
    from duck_detector.torchsetup import missing, prepare_cuda

    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - depends on how the venv was synced
        raise SystemExit(missing("ultralytics")) from error

    # Before the model is built, because this is what decides whether a convolution runs at all on
    # this machine — see the module for why. Ultralytics takes `device` as a string or index.
    device = args.device or prepare_cuda()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        # No `project=`: ultralytics resolves a relative one against its own settings root, which
        # turned `runs` into `runs/detect/runs/smoke`. Its default is `runs/detect/<name>`, and the
        # trainer says where it actually wrote — which is what the summary follows.
        name=args.name,
        exist_ok=True,
        # The camera rolls with the gait, so rotation is the augmentation that matters. Translation
        # and scale stay modest: this is one camera at one height, not a photo set.
        degrees=12.0,
        translate=0.08,
        scale=0.4,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        # Rooms differ in light far more than in colour balance, and the ISP handles white balance
        # before we see a frame.
        hsv_h=0.010,
        hsv_s=0.4,
        hsv_v=0.4,
        mosaic=1.0,
        close_mosaic=15,
        erasing=0.2,
        patience=40,
        seed=0,
        plots=True,
    )

    metrics = model.val(data=str(args.data), imgsz=args.imgsz, device=device)
    summary = {
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "imgsz": args.imgsz,
        "model": args.model,
        "smoke": "SMOKE" in header,
    }
    run = Path(model.trainer.save_dir)
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["run"] = str(run)
    print(json.dumps(summary, indent=2))

    if args.export:
        # Static shapes and a conservative opset, because the next step is `rknn-toolkit2` and it
        # accepts neither a dynamic batch nor the newest operators. `simplify` folds the shape
        # arithmetic that otherwise becomes unsupported ops on the NPU.
        path = model.export(
            format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False, nms=False
        )
        print(f"onnx → {path}")
        print(
            "next: convert on a machine with rknn-toolkit2 (RK3566 target, INT8, calibration set\n"
            "drawn from datasets/raw), then measure it on the board — quantisation is where a\n"
            "detector that worked on the laptop stops working."
        )


if __name__ == "__main__":
    main()
