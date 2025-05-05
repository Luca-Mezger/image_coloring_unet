import argparse
import io
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


# ---------------------------------------------------------------------------
#  Original code –‑ UNCHANGED
# ---------------------------------------------------------------------------

def load_checkpoint(state, filepath):
    with open(filepath, "rb") as f:
        state = serialization.from_bytes(state, f.read())
    return state


def preprocess_grayscale_image(image, img_size):
    # normalise to [0,1]
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    # ensure 3 channels
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    # centre‑crop to square (keeps aspect, no stretch)
    h, w = image.shape[:2]
    if h != w:
        if h > w:                                # portrait
            top = (h - w) // 2
            image = image[top:top+w, :, :]
        else:                                    # landscape
            left = (w - h) // 2
            image = image[:, left:left+h, :]

    # scale to model size
    if image.shape[0] != img_size:
        image = resize(image, (img_size, img_size), anti_aliasing=True)

    # Lab conversion → L channel
    lab = color.rgb2lab(image)
    L   = lab[..., 0]
    L_norm = (L / 50.0) - 1.0
    return L_norm, L


def colorize_image(state, image, img_size):
    """
    Predict ab channels from L and reconstruct colour image.
    Returns (L_orig, rgb_image)
    """
    L_norm, L_orig = preprocess_grayscale_image(image, img_size)
    L_norm_batch   = jnp.array(L_norm)[None, ..., None]
    model = create_model()
    pred_ab = model.apply({'params': state.params}, L_norm_batch)
    pred_ab = np.array(jnp.squeeze(pred_ab, axis=0))          # (H, W, 2)
    pred_ab_denorm = pred_ab * 128.0
    lab_image = np.concatenate([L_orig[..., None], pred_ab_denorm], axis=-1)
    rgb_image = color.lab2rgb(lab_image)
    return L_orig, rgb_image


def visualize(L_channel, colorized_rgb):
    """Side‑by‑side matplotlib visualisation."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(L_channel / 100.0, cmap="gray")
    axes[0].set_title("Input L Channel")
    axes[1].imshow(colorized_rgb)
    axes[1].set_title("Colorized")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using the trained UNet model to predict ab channels from the L channel.",
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/model.pkl",
                        help="Path to the saved model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input image (grayscale or color)")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image size used during training")
    args = parser.parse_args()

    # dummy init so Flax can create the parameter tree
    model = create_model()
    dummy = jnp.ones((1, args.img_size, args.img_size, 1))
    params = model.init(jax.random.PRNGKey(0), dummy)["params"]
    state  = TrainState(step=0, apply_fn=model.apply,
                        params=params, tx=None, opt_state=None)
    state  = load_checkpoint(state, args.checkpoint)
    print(f"✓ loaded checkpoint  {args.checkpoint}")

    img = imageio.imread(args.input)
    L, rgb = colorize_image(state, img, args.img_size)
    visualize(L, rgb)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
#  EXTRA HELPERS  (for the Flask backend)  – added below the CLI guard
# ---------------------------------------------------------------------------

_DEFAULT_CKPT      = "checkpoints/model_1_lr=0.0003_epochs=10_bs=2.pkl" 
_DEFAULT_IMG_SIZE  = 256


def load_model_state(img_size: int = _DEFAULT_IMG_SIZE,
                     ckpt: str = _DEFAULT_CKPT) -> TrainState:
    """
    One‑time model loader for the web server.
    """
    model  = create_model()
    dummy  = jnp.ones((1, img_size, img_size, 1))
    params = model.init(jax.random.PRNGKey(0), dummy)["params"]
    state  = TrainState(step=0, apply_fn=model.apply,
                        params=params, tx=None, opt_state=None)
    return load_checkpoint(state, ckpt)


def colorize_bytes(state: TrainState,
                   img_bytes: bytes,
                   img_size: int = _DEFAULT_IMG_SIZE) -> bytes:
    """
    Accept raw image bytes → return PNG bytes of the colourised result.
    Suitable for Flask's send_file().
    """
    img = imageio.imread(io.BytesIO(img_bytes))
    _, rgb = colorize_image(state, img, img_size)
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    buf = io.BytesIO()
    imageio.imwrite(buf, rgb_uint8, format="png")
    buf.seek(0)
    return buf.getvalue()
