#!/usr/bin/env python3
import os
import random
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from skimage import color, io
from skimage.transform import resize
from model import create_model
from train import TrainState

# ASCII shades from dark to light
_ASCII_CHARS = "@%#*+=-:. "

def load_checkpoint(state, path):
    with open(path, "rb") as f:
        return serialization.from_bytes(state, f.read())

def preprocess(image, img_size):
    # normalize & ensure 3‑channel
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    if image.shape[:2] != (img_size, img_size):
        image = resize(image, (img_size, img_size), anti_aliasing=True)
    lab = color.rgb2lab(image)
    L, ab = lab[...,0], lab[...,1:]
    L_norm = (L/50.0) - 1.0
    ab_norm = ab / 128.0
    return L_norm, ab_norm, L, ab

def predict_ab(model, params, L_norm):
    inp = jnp.array(L_norm)[None,...,None]
    pred = model.apply({'params': params}, inp)
    return np.squeeze(np.array(pred), 0) * 128.0  # back to ab-scale

def to_ascii_lines(gray2d, width=32):
    # scale to [0,255]
    p = gray2d.astype(np.float32)
    p = (p - p.min()) / max(p.max()-p.min(), 1e-6) * 255
    # resize to square block
    block = resize(p, (width, width), anti_aliasing=True)
    chars = _ASCII_CHARS
    m = len(chars)-1
    lines = []
    for row in block:
        line = "".join(chars[int(val/255*m)] for val in row)
        lines.append(line)
    return lines

def rgb_to_gray(rgb):
    # simple luminance
    return 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data/images")
    p.add_argument("--checkpoint", default="checkpoints/new_model.pkl")
    p.add_argument("--img_size",    type=int, default=64)
    args = p.parse_args()

    # init model & state
    model = create_model()
    dummy = jnp.ones((1, args.img_size, args.img_size, 1))
    params = model.init(jax.random.PRNGKey(0), dummy)["params"]
    state = TrainState(step=0, apply_fn=model.apply,
                       params=params, tx=None, opt_state=None)
    state = load_checkpoint(state, args.checkpoint)

    # pick 10 random files
    files = [f for f in os.listdir(args.data_dir)
             if f.lower().endswith((".jpg",".jpeg",".png"))]
    pick = random.sample(files, k=10)

    for fname in pick:
        path = os.path.join(args.data_dir, fname)
        img  = io.imread(path)
        Ln, abn, L, ab = preprocess(img, args.img_size)
        pred_ab = predict_ab(model, state.params, Ln)

        # raw RGBs
        gt_rgb   = np.clip(color.lab2rgb(np.stack([L,ab[...,0],ab[...,1]],-1)),0,1)
        pred_rgb = np.clip(color.lab2rgb(np.stack([L,pred_ab[...,0],pred_ab[...,1]],-1)),0,1)

        # create grayscale brightness maps
        gray_in  = L/100.0       # L channel [0,100] → [0,1]
        gray_gt  = rgb_to_gray(gt_rgb)
        gray_pred= rgb_to_gray(pred_rgb)

        # ascii
        A_in   = to_ascii_lines(gray_in)
        A_gt   = to_ascii_lines(gray_gt)
        A_pred = to_ascii_lines(gray_pred)

        print(f"\n=== {fname} ===\n")
        for row_in, row_gt, row_pr in zip(A_in, A_gt, A_pred):
            # side-by-side with a space between
            print(f"{row_in}  {row_gt}  {row_pr}")
    print()

if __name__ == "__main__":
    main()
