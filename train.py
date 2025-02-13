import os
import math
import jax
import jax.numpy as jnp
import optax
import argparse
from flax.training import train_state
from model import create_model
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    valid_extensions = ('.jpg', '.jpeg', '.png')
    file_list = [
        f for f in os.listdir('data/images')
        if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join('data/images', f))
    ]
    total_batches = math.ceil(len(file_list) / config['batch_size'])

    dataset = load_dataset(batch_size=config['batch_size'])
    
    # Initialize model & training state
    model = create_model()
    rng = jax.random.PRNGKey(config['seed'])
    rng, init_rng = jax.random.split(rng)
    input_shape = (1, config['img_size'], config['img_size'], 1)
    params = model.init(init_rng, jnp.ones(input_shape))['params']
    tx = optax.adam(config['learning_rate'])
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    epoch_losses = []
    for epoch in range(1, config['epochs'] + 1):
        epoch_loss, count = 0.0, 0
        for batch in tqdm(dataset.take(total_batches), desc=f"Epoch {epoch}", total=total_batches, unit="batch"):
            # Efficient conversion to JAX arrays
            batch_np = (jnp.asarray(batch[0]), jnp.asarray(batch[1]))
            
            state, loss = train_step(state, batch_np)
            epoch_loss += float(loss)
            count += 1
        avg_loss = epoch_loss / count
        epoch_losses.append(avg_loss)
        print(f"\nEpoch {epoch} | Average Loss: {avg_loss:.4f}")

    # Save loss plot instead of blocking execution
    plt.plot(range(1, config['epochs'] + 1), epoch_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Performance")
    plt.savefig("training_loss.png")

    return state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config = {
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'seed': args.seed
    }

    train(config)
