import os

# Optional JAX / TF memory tweaks
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
# os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
# os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

import argparse
from train import train
from eval import evaluate
from utils import save_checkpoint, save_metrics
from config import config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the JAX UNet colourisation model with custom hyper‑parameters."
    )
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--img_size", type=int, help="Square image size")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--checkpoint_path", type=str, help="Checkpoint filename")

    args = parser.parse_args()

    # -------- override defaults when flags are provided -------- #
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.img_size is not None:
        config["img_size"] = args.img_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.seed is not None:
        config["seed"] = args.seed

    # -------- checkpoint path -------- #
    if args.checkpoint_path:
        config["checkpoint_path"] = args.checkpoint_path
    else:
        cfg = config
        config["checkpoint_path"] = (
            f"{cfg['checkpoint_path']}_img{cfg['img_size']}"
            f"_lr{cfg['learning_rate']}_bs{cfg['batch_size']}"
            f"_epochs{cfg['epochs']}.pkl"
        )

    # -------- train, evaluate, save artefacts -------- #
    state, metrics = train(config)
    evaluate(state, config)

    ckpt_path = config["checkpoint_path"]
    save_checkpoint(state, ckpt_path)
    save_metrics(metrics, ckpt_path)


if __name__ == "__main__":
    main()
