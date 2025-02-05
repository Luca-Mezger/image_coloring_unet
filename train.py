import os
import math
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from model import create_model
from datasets import load_dataset  # Your dataset loader
from tqdm import tqdm

class TrainState(train_state.TrainState):
    pass

def loss_fn(params, apply_fn, batch):
    grayscale, target = batch
    pred = apply_fn({'params': params}, grayscale)
    return jnp.mean(jnp.abs(pred - target))

@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, state.apply_fn, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train(config):
    # Determine total number of batches per epoch based on number of files
    valid_extensions = ('.jpg', '.jpeg', '.png')
    file_list = [
        f for f in os.listdir('data/images')
        if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join('data/images', f))
    ]
    num_files = len(file_list)
    total_batches = math.ceil(num_files / config['batch_size'])
    
    # Repeat dataset indefinitely and take fixed batches per epoch.
    dataset = load_dataset(batch_size=config['batch_size']).repeat()
    model = create_model()
    rng = jax.random.PRNGKey(config['seed'])
    input_shape = (1, config['img_size'], config['img_size'], 1)
    params = model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adam(config['learning_rate'])
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    for epoch in range(1, config['epochs'] + 1):
        epoch_loss = 0.0
        count = 0
        for batch in tqdm(dataset.take(total_batches), desc=f"Epoch {epoch}", total=total_batches, unit="batch"):
            # Convert TF tensors to NumPy, then to JAX arrays.
            grayscale_np = batch[0].numpy()
            color_np = batch[1].numpy()
            batch_np = (jnp.array(grayscale_np), jnp.array(color_np))
            
            state, loss = train_step(state, batch_np)
            loss_value = float(loss)  # Convert loss to a Python float.
            epoch_loss += loss_value
            count += 1
        avg_loss = epoch_loss / count
        print(f"\nEpoch {epoch} | Average Loss: {avg_loss:.4f}")
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
