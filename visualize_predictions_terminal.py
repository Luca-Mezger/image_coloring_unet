#!/usr/bin/env python3
import os
import random
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from model import create_model
from train import TrainState
from skimage import color, io
from skimage.transform import resize
from PIL import Image as PILImage
from rich.console import Console
from rich.columns import Columns
from rich.image import Image as RichImage

def load_checkpoint(state, filepath):
    with open(filepath, "rb") as f:
        state = serialization.from_bytes(state, f.read())
    return state

def preprocess_image(image, img_size):
    # normalize & resize
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[:2] != (img_size, img_size):
        image = resize(image, (img_size, img_size), anti_aliasing=True)
    # to Lab
    lab = color.rgb2lab(image)
    L = lab[..., 0]
    ab = lab[..., 1:]
    L_norm = (L / 50.0) - 1.0
    ab_norm = ab / 128.0
    return L_norm, ab_norm, L, ab

def colorize(model, params, L_norm):
    L_batch = jnp.array(L_norm)[None, ..., None]
    pred_ab = model.apply({'params': params}, L_batch)
    pred_ab = np.squeeze(np.array(pred_ab), axis=0) * 128.0
    return pred_ab

def to_pil_gray(L):
    # L in [0,100] → scale to [0,255]
    gray = np.clip((L / 100.0 * 255), 0, 255).astype(np.uint8)
    return PILImage.fromarray(gray, mode="L")

def to_pil_rgb(L, ab):
    lab = np.concatenate([L[..., None], ab], axis=-1)
    rgb = np.clip(color.lab2rgb(lab), 0, 1)
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    return PILImage.fromarray(rgb_uint8, mode="RGB")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data/images")
    p.add_argument("--checkpoint", default="checkpoints/new_model.pkl")
    p.add_argument("--img_size",    type=int, default=64)
    args = p.parse_args()

    # init model + state
    model = create_model()
    dummy = jnp.ones((1, args.img_size, args.img_size, 1))
    params = model.init(jax.random.PRNGKey(0), dummy)["params"]
    state = TrainState(step=0, apply_fn=model.apply, params=params, tx=None, opt_state=None)
    state = load_checkpoint(state, args.checkpoint)

    # pick 10 random files
    imgs = [
        f for f in os.listdir(args.data_dir)
        if f.lower().endswith((".jpg",".jpeg",".png"))
    ]
    selection = random.sample(imgs, k=10)

    console = Console()
    panels = []
    for fname in selection:
        path = os.path.join(args.data_dir, fname)
        raw = io.imread(path)
        L_norm, ab_norm, L_orig, ab_orig = preprocess_image(raw, args.img_size)
        pred_ab = colorize(model, state.params, L_norm)

        pil_gray      = to_pil_gray(L_orig)
        pil_gt        = to_pil_rgb(L_orig, ab_orig)
        pil_pred      = to_pil_rgb(L_orig, pred_ab)

        # wrap with RichImage
        rich_gray = RichImage.from_pil(pil_gray, width=20)
        rich_gt   = RichImage.from_pil(pil_gt,   width=20)
        rich_pred = RichImage.from_pil(pil_pred, width=20)

        panels.append(Columns([rich_gray, rich_gt, rich_pred], equal=True, expand=True))

    # print filename header + images
    for fname, col in zip(selection, panels):
        console.rule(f"[bold]{fname}")
        console.print(col)

if __name__ == "__main__":
    main()
