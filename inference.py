import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from model import create_model
import imageio.v2 as imageio  # Use imageio.v2 to avoid deprecation warnings
import numpy as np
from train import TrainState
from flax import serialization
from skimage.transform import resize

def load_checkpoint(state, filepath):
    with open(filepath, "rb") as f:
        state = serialization.from_bytes(state, f.read())
    return state

def colorize_image(state, grayscale_image):
    """
    Colorize a single grayscale image.
    
    Args:
        state: Trained model state.
        grayscale_image: Array of shape (H, W, 1) or (1, H, W, 1).
    Returns:
        Colorized image of shape (H, W, 3).
    """
    model = create_model()
    # Ensure the image has a batch dimension.
    if grayscale_image.ndim == 3:
        grayscale_image = jnp.expand_dims(grayscale_image, axis=0)
    pred = model.apply({'params': state.params}, grayscale_image)
    return jnp.squeeze(pred, axis=0)

def visualize(grayscale, colorized):
    """
    Display grayscale and colorized images side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(grayscale.squeeze(), cmap='gray')
    axes[0].set_title('Grayscale')
    # Assuming tanh activation outputs values in [-1, 1], rescale to [0, 1].
    axes[1].imshow((colorized + 1) / 2)
    axes[1].set_title('Colorized')
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a grayscale image using the trained UNet model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model.pkl",
        help="Path to the saved model checkpoint"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input image (grayscale or color)"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Image size used during training"
    )
    args = parser.parse_args()

    # Initialize the model with dummy input so parameters can be created.
    model = create_model()
    dummy_input = jnp.ones((1, args.img_size, args.img_size, 1))
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_input)['params']
    print("Initialized dummy model parameters")
    # Create a TrainState with the initialized parameters.
    state = TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=None,
        opt_state=None
    )
    state = load_checkpoint(state, args.checkpoint)
    print(f"Loaded model checkpoint from {args.checkpoint}")
    # Load the input image.
    img = imageio.imread(args.input)
    print(f"Loaded image with shape {img.shape}")
    # Convert to grayscale if necessary.
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    print(f"Converted to grayscale")
    # Ensure shape is (H, W, 1).
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    # Normalize the image to [0, 1] if it's not already float.
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    # Resize if needed.
    if img.shape[0] != args.img_size or img.shape[1] != args.img_size:
        img = resize(img, (args.img_size, args.img_size), anti_aliasing=True)
    grayscale_image = jnp.array(img, dtype=jnp.float32)

    print(f"Processed image with shape {grayscale_image.shape}")

    # Run inference.
    colorized = colorize_image(state, grayscale_image)
    print(f"Colorized image with shape {colorized.shape}")
    visualize(grayscale_image, colorized)

if __name__ == "__main__":
    main()
