import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from model import create_model
from datasets import load_dataset  # Implement your own dataset loader

class TrainState(train_state.TrainState):
    """Training state for the UNet model."""
    pass

def loss_fn(params, apply_fn, batch):
    """
    Compute L1 loss between predicted color image and target image.
    batch: tuple (grayscale, target) where
        grayscale: (B, H, W, 1)
        target: (B, H, W, 3)
    """
    grayscale, target = batch
    pred = apply_fn({'params': params}, grayscale)
    return jnp.mean(jnp.abs(pred - target))

@jax.jit
def train_step(state, batch):
    """Perform one training step."""
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, state.apply_fn, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train(config):
    """
    Training loop for the UNet model.
    
    Args:
        config: Dictionary with training configurations.
    Returns:
        state: The trained model state.
    """
    # Load dataset (should yield batches of (grayscale, color) images)
    dataset = load_dataset(batch_size=config['batch_size'])
    
    # Create model and initialize parameters
    model = create_model()
    rng = jax.random.PRNGKey(config['seed'])
    # Assume grayscale images of shape (img_size, img_size, 1)
    input_shape = (1, config['img_size'], config['img_size'], 1)
    params = model.init(rng, jnp.ones(input_shape))['params']
    
    # Setup optimizer using Adam
    tx = optax.adam(config['learning_rate'])
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        epoch_loss = 0.0
        count = 0
        for batch in dataset:
            state, loss = train_step(state, batch)
            epoch_loss += loss
            count += 1
        avg_loss = epoch_loss / count
        print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")
    return state

if __name__ == "__main__":
    config = {
        'batch_size': 32,
        'img_size': 256,
        'epochs': 10,
        'learning_rate': 1e-3,
        'seed': 42
    }
    train(config)
