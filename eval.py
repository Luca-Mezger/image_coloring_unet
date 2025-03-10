import jax.numpy as jnp
from model import create_model
from datasets import load_dataset  # Dataset loader

def evaluate(state, config):
    """
    Evaluate the UNet model for Lab colorization using L1 loss.
    
    The dataset yields (L, ab) pairs:
      - L: normalized L channel with shape (H, W, 1) in [-1, 1]
      - ab: normalized ab channels with shape (H, W, 2) in [-1, 1]
    
    Args:
        state: Trained model state with parameters.
        config: Dictionary with evaluation settings.
    
    Returns:
        Average L1 loss over the evaluation dataset.
    """
    dataset = load_dataset(batch_size=config['batch_size'])
    model = create_model()
    total_loss = 0.0
    count = 0
    for batch in dataset:
        # Convert TensorFlow tensors to numpy arrays then to JAX arrays.
        L_np = batch[0].numpy()    # L channel input
        ab_np = batch[1].numpy()   # Target ab channels
        L_jax = jnp.array(L_np)
        ab_jax = jnp.array(ab_np)
        
        pred_ab = model.apply({'params': state.params}, L_jax)
        loss = jnp.mean(jnp.abs(pred_ab - ab_jax))
        total_loss += float(loss)
        count += 1
    avg_loss = total_loss / count
    print(f"Average Evaluation Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    from train import TrainState
    dummy_state = TrainState(step=0, apply_fn=create_model().apply, params={}, tx=None)
    config = {'batch_size': 32, 'img_size': 256}
    evaluate(dummy_state, config)
