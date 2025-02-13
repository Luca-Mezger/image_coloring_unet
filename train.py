import os
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from flax.training import train_state
from jax.experimental import checkpoint

from data_loader import load_dataset
from model import UNet

# Limit JAX to 80% of GPU memory to avoid out-of-memory errors
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# Enable mixed precision to reduce VRAM usage
jax.config.update("jax_default_matmul_precision", "bfloat16")


def create_train_state(rng, learning_rate):
    """Initialize training state with model parameters."""
    model = UNet()
    rng, init_rng = jax.random.split(rng)
    
    # Create initial model parameters
    params = model.init(init_rng, jnp.ones((1, 512, 512, 1)))['params']
    optimizer = optax.adam(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )


# Fix PRNG key handling to avoid reusing the same key
rng = jax.random.PRNGKey(42)
rng, train_rng = jax.random.split(rng)


@jax.jit
def train_step(state, batch):
    """Single training step including forward pass, loss calculation, and backpropagation."""

    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, batch[0])
        return jnp.mean((predictions - batch[1]) ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss


def train(config):
    """Main training loop."""
    dataset = load_dataset(batch_size=config['batch_size']).shuffle(1000).repeat()
    
    # Convert dataset batches to JAX device arrays
    dataset = ((jax.device_put(batch[0]), jax.device_put(batch[1])) for batch in dataset)
    
    state = create_train_state(train_rng, config['learning_rate'])
    
    losses = []
    
    for epoch in range(config['epochs']):
        epoch_loss = 0
        
        for batch in dataset:
            state, loss = train_step(state, batch)
            epoch_loss += loss
        
        avg_loss = epoch_loss / config['batch_size']
        losses.append(avg_loss)
        
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    
    return state, losses


if __name__ == "__main__":
    config = {
        'batch_size': 4,
        'img_size': 512,
        'epochs': 10,
        'learning_rate': 0.0005
    }

    final_state, loss_history = train(config)

    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.savefig("training_loss.png")
