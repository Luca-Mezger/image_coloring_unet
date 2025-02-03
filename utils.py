import os
import pickle

def save_checkpoint(state, filepath):
    """Save model state (parameters, optimizer state, etc.) to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)
    print(f"Checkpoint saved at {filepath}")

def load_checkpoint(filepath):
    """Load model state from a checkpoint file."""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        print(f"Checkpoint loaded from {filepath}")
        return state
    else:
        print(f"No checkpoint found at {filepath}")
        return None
