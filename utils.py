import os
from flax import serialization

def save_checkpoint(state, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(serialization.to_bytes(state))
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(state, filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            state = serialization.from_bytes(state, f.read())
        print(f"Checkpoint loaded from {filepath}")
        return state
    else:
        print(f"No checkpoint found at {filepath}")
        return state
