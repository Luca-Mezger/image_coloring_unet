import jax.numpy as jnp
import matplotlib.pyplot as plt
from model import create_model

def colorize_image(state, grayscale_image):
    """
    Colorize a single grayscale image.
    
    Args:
        state: Trained model state containing parameters.
        grayscale_image: Array of shape (H, W, 1) or (1, H, W, 1).
    Returns:
        Colorized image of shape (H, W, 3).
    """
    model = create_model()
    # Ensure the image has a batch dimension.
    if grayscale_image.ndim == 3:
        grayscale_image = jnp.expand_dims(grayscale_image, axis=0)
    pred = model.apply({'params': state.params}, grayscale_image)
    # Remove the batch dimension.
    return jnp.squeeze(pred, axis=0)

def visualize(grayscale, colorized):
    """
    Display grayscale and colorized images side by side.
    
    Args:
        grayscale: Grayscale image.
        colorized: Colorized image (output of tanh, values in [-1,1]).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(grayscale.squeeze(), cmap='gray')
    axes[0].set_title('Grayscale')
    # Scale tanh output from [-1, 1] to [0, 1]
    axes[1].imshow((colorized + 1) / 2)
    axes[1].set_title('Colorized')
    plt.show()

if __name__ == "__main__":
    import numpy as np
    from train import TrainState
    # Dummy state; replace with your trained state.
    dummy_params = {}
    state = TrainState(step=0, apply_fn=create_model().apply, params=dummy_params, tx=None)
    # Dummy grayscale image
    grayscale_image = jnp.array(np.random.rand(256, 256, 1), dtype=jnp.float32)
    colorized = colorize_image(state, grayscale_image)
    visualize(grayscale_image, colorized)
