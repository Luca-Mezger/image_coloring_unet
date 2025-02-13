import os

# Reduce JAX GPU memory usage
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable preallocation
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"  # Use only 50% of available GPU memory
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"  # Allows memory sharing between CPU/GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

import argparse
from train import train
from eval import evaluate
from utils import save_checkpoint
from config import config

def main():
    parser = argparse.ArgumentParser(
        description="Train the JAX image colorization model with custom hyperparameters."
    )
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--img_size", type=int, help="Image height/width (images are square)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for the optimizer")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--checkpoint_path", type=str, help="Path to save the model checkpoint")

    args = parser.parse_args()

    # Override config values with command-line arguments if provided.
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.img_size is not None:
        config['img_size'] = args.img_size
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.seed is not None:
        config['seed'] = args.seed
    if args.checkpoint_path is not None:
        config['checkpoint_path'] = args.checkpoint_path
    else:
        config['checkpoint_path'] = f"{config['checkpoint_path']}_img{config['img_size']}_lr{config['learning_rate']}_bs{config['batch_size']}_epochs{config['epochs']}.pkl"


    # Train, evaluate, and save the model.
    state = train(config)
    evaluate(state, config)
    save_checkpoint(state, config['checkpoint_path'])

if __name__ == "__main__":
    main()
