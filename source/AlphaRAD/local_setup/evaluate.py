"""Run the original AlphaRAD evaluation with a complete local YAML config."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "local_setup/eval_local.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dataset", action="append", help="Evaluate only this configured dataset; repeat as needed.")
    parser.add_argument("--check-only", action="store_true", help="Validate configuration and imports without running the full evaluation.")
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / "alpharad-assets/.cache/matplotlib"))
    import torch
    from omegaconf import OmegaConf
    from evaluate_zeroshot import main as upstream_main

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu or an accessible GPU.")
    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)
    if args.dataset:
        missing = set(args.dataset) - set(config.data_test)
        if missing:
            raise ValueError(f"Datasets not present in configuration: {sorted(missing)}")
        for name in list(config.data_test):
            if name not in args.dataset:
                del config.data_test[name]
    for path in config.test.checkpoint:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    print(f"Config: {args.config.resolve()}; datasets: {list(config.data_test)}; device: {args.device}")
    if args.check_only:
        print("Configuration and official evaluation imports validated. No evaluation was run.")
        return
    # The underlying function is the author's evaluation; bypass Hydra's CLI so
    # an already composed local YAML can be passed without changing repo configs.
    upstream_main.__wrapped__(config)


if __name__ == "__main__":
    main()
