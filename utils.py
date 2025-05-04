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
    
def save_metrics(metrics: dict, checkpoint_path: str) -> None:
    """
    Write metrics to '<checkpoint_stem>_metrics.txt' next to the .pkl file.
    Lists are serialized as comma‑separated values.
    """
    txt_path = os.path.splitext(checkpoint_path)[0] + "_metrics.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            if isinstance(v, list):
                v = ",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in v)
            f.write(f"{k}: {v}\n")
    print(f"Metrics saved to {txt_path}")