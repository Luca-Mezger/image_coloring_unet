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
        # Convert TensorFlow tensors to numpy arrays then to JAX arrays.
        grayscale_np = batch[0].numpy()
        target_np = batch[1].numpy()
        grayscale_jax = jnp.array(grayscale_np)
        target_jax = jnp.array(target_np)
        
        pred = model.apply({'params': state.params}, grayscale_jax)
        loss = jnp.mean(jnp.abs(pred - target_jax))
        total_loss += float(loss)
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
