"""The trained detector, to and from the Hub.

    uv run model push runs/detect/duck-v1          # tagged duck-v1; main points at it
    uv run model push runs/detect/duck-v1 --tag duck-v1-int8   # a different name, if wanted
    uv run model pull                              # weights/duck_detect.{pt,onnx,rknn} from main
    uv run model pull --revision duck-v1

`train --push` does the first of these when training finishes; `scripts/to_rknn.py` says to run it
again once the `.rknn` exists, which adds that file under the same tag. See `hub.py` for the layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from duck_detector import hub


def main() -> None:
    parser = argparse.ArgumentParser(description="Push or pull the detector's weights on the Hub.")
    parser.add_argument("--repo", default=hub.MODEL_REPO)
    sub = parser.add_subparsers(dest="command", required=True)

    push = sub.add_parser("push", help="upload a run's weights, tagged with the run's name")
    push.add_argument("run", type=Path, help="a runs/detect/<name> directory")
    push.add_argument("--tag", help="default: the run directory's name")
    push.add_argument("--public", action="store_true", help="create the repo public, if creating")

    pull = sub.add_parser("pull", help="download the weights into weights/")
    pull.add_argument("--revision", help="a tag or commit; default main")
    pull.add_argument("--into", type=Path, default=hub.WEIGHTS)

    args = parser.parse_args()
    if args.command == "push":
        hub.push_model(args.run, tag=args.tag, repo=args.repo, private=not args.public)
    else:
        hub.pull_model(revision=args.revision, repo=args.repo, into=args.into)


if __name__ == "__main__":
    main()
