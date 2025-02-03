import jax.numpy as jnp
from model import create_model
from datasets import load_dataset  # Your dataset loader

def evaluate(state, config):
    """
    Evaluate the UNet model using L1 loss.
    
    Args:
        state: Trained model state with parameters.
        config: Dictionary with evaluation settings.
    """
    dataset = load_dataset(batch_size=config['batch_size'])
    model = create_model()
    total_loss = 0.0
    count = 0
    for batch in dataset:
        grayscale, target = batch
        pred = model.apply({'params': state.params}, grayscale)
        loss = jnp.mean(jnp.abs(pred - target))
        total_loss += loss
        count += 1
    avg_loss = total_loss / count
    print(f"Average Evaluation Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    # Replace with actual state for evaluation.
    from train import TrainState
    dummy_state = TrainState(step=0, apply_fn=create_model().apply, params={}, tx=None)
    config = {'batch_size': 32, 'img_size': 256}
    evaluate(dummy_state, config)
