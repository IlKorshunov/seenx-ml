from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass(frozen=True)
class Experiment:
    module: str
    supports_args: bool = True


EXPERIMENTS: dict[str, Experiment] = {
    "bert.seq": Experiment("train.bert.train_bert_seq"),
    "clustering.content_specialists": Experiment("train.clustering.content_cluster_specialists"),
    "clustering.multimodal_specialists": Experiment("train.clustering.cluster_specialists_multimodal"),
    "lstm.seq": Experiment("train.lstm.train_lstm_seq"),
    "metamodel.train": Experiment("train.metamodel.train_metamodel"),
    "transformer.multimodal_seq": Experiment("train.transformer.train_multimodal_seq"),
    "videomae.seq": Experiment("train.videomae.train_videomae_seq"),
}

def dispatch(key: str, forwarded_args: list[str]) -> None:
    if key not in EXPERIMENTS:
        print(f"Unknown experiment: {key}", file=sys.stderr)
        print("Run `python -m train.run --list` to see available experiments", file=sys.stderr)
        raise SystemExit(2)

    entry = EXPERIMENTS[key]
    if forwarded_args and not entry.supports_args:
        print(f"Warning: {key} does not define CLI args")

    selected_main = getattr(importlib.import_module(entry.module), "main", None)
    if selected_main is None:
        raise SystemExit(f"Experiment module has no main(): {entry.module}")

    old_argv = sys.argv
    sys.argv = [f"train.run {key}", *forwarded_args]
    try:
        selected_main()
    finally:
        sys.argv = old_argv


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] == "--help":
        print("python -m train.run <experiment> [args]")
        return
    if args[0] == "--list":
        for key in sorted(EXPERIMENTS):
            print(f"{key:36} {EXPERIMENTS[key].module}")
        return
    dispatch(args[0], args[1:])

if __name__ == "__main__":
    main()
