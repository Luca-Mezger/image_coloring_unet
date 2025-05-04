#!/usr/bin/env python3
"""
Download images from a torchvision dataset **only if they are not already
present** in the target folder. Images are saved as
   <save_folder>/image_<N>.png
where N starts at 0.
"""

import argparse
import os
import random
from pathlib import Path

from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset

# --------------------------------------------------------------------------- #
def get_dataset(name: str, root: str, img_size: int):
    """Return a torchvision dataset instance without triggering a re‑download
    when the compressed tar/zip is already present."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # `download=` will do nothing if the archive has already been fetched;
    # so we leave it `True` to handle the first‑run case automatically.
    if name == "STL10":
        return datasets.STL10(root=root, split="train", download=True, transform=transform)
    if name == "CelebA":
        return datasets.CelebA(root=root, split="train", download=True, transform=transform)
    if name == "ImageNet":
        # Manual download required; path must point to the extracted ILSVRC folder
        return datasets.ImageFolder(root=root, transform=transform)
    raise ValueError("Unsupported dataset. Choose from STL10, CelebA, or ImageNet.")
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and save images from a torchvision dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="STL10",
                        choices=["STL10", "CelebA", "ImageNet"],
                        help="Dataset to use.")
    parser.add_argument("--save_folder", type=str, default="./data/images",
                        help="Folder to save PNG images.")
    parser.add_argument("--num_images", type=int, default=3000,
                        help="Number of images to have locally (0 = all).")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Output resolution (min 32).")

    args = parser.parse_args()

    DATASET_NAME = args.dataset
    SAVE_FOLDER = Path(args.save_folder)
    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1) Count existing PNG/JPG images
    existing_files = sorted([p for p in SAVE_FOLDER.glob("image_*.png")])
    already_have = len(existing_files)

    # If we already have enough images, we are done.
    target_total = None if args.num_images <= 0 else args.num_images
    if target_total is not None and already_have >= target_total:
        print(f"{already_have} images already present in '{SAVE_FOLDER}'. Nothing to do.")
        return
    # ------------------------------------------------------------------ #

    # 2) Load dataset (downloads archives only if absent)
    dataset_root = "./data" if DATASET_NAME != "ImageNet" else SAVE_FOLDER  # arbitrary
    dataset = get_dataset(DATASET_NAME, dataset_root, args.img_size)

    # 3) Optionally select a random subset
    if target_total is not None:
        remaining_to_save = target_total - already_have
        indices = random.sample(range(len(dataset)), remaining_to_save)
        dataset = Subset(dataset, indices)
    else:
        remaining_to_save = len(dataset)  # all

    # 4) Save only the missing images
    print(f"Saving {remaining_to_save} new image(s) to '{SAVE_FOLDER}' ...")
    next_index = already_have  # continue numbering
    to_pil = transforms.ToPILImage()

    for image_tensor, _ in dataset:
        out_path = SAVE_FOLDER / f"image_{next_index}.png"
        if out_path.exists():          # safety check
            next_index += 1
            continue
        img: Image.Image = to_pil(image_tensor)  # type: ignore
        img.save(out_path)
        next_index += 1

    final_count = len(list(SAVE_FOLDER.glob('image_*.png')))
    print(f"Now have {final_count} images in '{SAVE_FOLDER}'. Done.")


if __name__ == "__main__":
    main()
