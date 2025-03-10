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
from skimage import color

def load_checkpoint(state, filepath):
    with open(filepath, "rb") as f:
        state = serialization.from_bytes(state, f.read())
    return state

def preprocess_grayscale_image(image, img_size):
    """
    Preprocess the input image:
    - Normalize to [0, 1].
    - If grayscale, replicate channels to form a pseudo-RGB image.
    - Resize to the target image size.
    - Convert to Lab and extract the L channel.
    - Normalize L to [-1, 1] as (L/50) - 1.
    
    Returns:
        L_norm: Normalized L channel (H, W) in [-1, 1].
        L_orig: Original L channel (H, W) from Lab.
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[0] != img_size or image.shape[1] != img_size:
        image = resize(image, (img_size, img_size), anti_aliasing=True)
    lab = color.rgb2lab(image)
    L = lab[..., 0]
    L_norm = (L / 50.0) - 1.0
    return L_norm, L

def colorize_image(state, image, img_size):
    """
    Colorize an image by predicting ab channels from the normalized L channel.
    
    Args:
        state: Trained model state.
        image: Input image array.
        img_size: Image size used during training.
    
    Returns:
        Tuple (L_orig, colorized_rgb): The original L channel and the reconstructed RGB image.
    """
    L_norm, L_orig = preprocess_grayscale_image(image, img_size)
    # Prepare input: add batch and channel dimensions.
    L_norm_batch = jnp.array(L_norm)[None, ..., None]
    model = create_model()
    pred_ab = model.apply({'params': state.params}, L_norm_batch)
    pred_ab = np.array(jnp.squeeze(pred_ab, axis=0))  # Shape: (H, W, 2)
    # Denormalize ab channels.
    pred_ab_denorm = pred_ab * 128.0
    # Combine the original L channel with the predicted ab channels.
    lab_image = np.concatenate([L_orig[..., None], pred_ab_denorm], axis=-1)
    # Convert the Lab image to RGB.
    rgb_image = color.lab2rgb(lab_image)
    return L_orig, rgb_image

def visualize(L_channel, colorized_rgb):
    """
    Display the input L channel and the colorized RGB image side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(L_channel / 100.0, cmap='gray')
    axes[0].set_title('Input L Channel')
    axes[1].imshow(colorized_rgb)
    axes[1].set_title('Colorized')
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Run inference using the trained UNet model to predict ab channels from the L channel."
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
    if img.ndim == 3 and img.shape[-1] == 3:
        print("Using color image to extract L channel.")
    else:
        print("Using grayscale image.")
    
    L_channel, colorized = colorize_image(state, img, args.img_size)
    print(f"Colorized image with shape {colorized.shape}")
    visualize(L_channel, colorized)

if __name__ == "__main__":
    main()
