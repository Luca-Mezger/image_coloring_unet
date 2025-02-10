import argparse
import os
import random
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset

def main():
    parser = argparse.ArgumentParser(
        description="Download and save images from a torchvision dataset."
    )
    parser.add_argument(
        "--dataset", type=str, default="STL10",
        choices=["STL10", "CelebA", "ImageNet"],
        help="Dataset to use: STL10, CelebA, or ImageNet."
    )
    parser.add_argument(
        "--save_folder", type=str, default="./data/images",
        help="Folder to save images."
    )
    parser.add_argument(
        "--num_images", type=int, default=3000,
        help="Number of images to download (set to 0 to download all)."
    )
    parser.add_argument(
        "--img_size", type=int, default=512,
        help="Image resolution. Images are resized to (img_size, img_size). Minimum size is 32."
    )

    args = parser.parse_args()

    DATASET_NAME = args.dataset
    SAVE_FOLDER = args.save_folder
    NUM_IMAGES = args.num_images if args.num_images > 0 else None
    IMG_SIZE = (args.img_size, args.img_size)

    # Create save folder if it doesn't exist
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    def get_dataset(name):
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor()
        ])
        if name == "STL10":
            return datasets.STL10(root=SAVE_FOLDER, download=True, split='train', transform=transform)
        elif name == "CelebA":
            return datasets.CelebA(root=SAVE_FOLDER, download=True, split='train', transform=transform)
        elif name == "ImageNet":
            return datasets.ImageFolder(root="/path/to/imagenet", transform=transform)  # Requires manual download
        else:
            raise ValueError("Unsupported dataset. Choose from STL10, CelebA, or ImageNet.")

    # Load the dataset
    dataset = get_dataset(DATASET_NAME)

    # Select a subset of images if NUM_IMAGES is specified
    if NUM_IMAGES is not None:
        indices = random.sample(range(len(dataset)), min(NUM_IMAGES, len(dataset)))
        dataset = Subset(dataset, indices)

    # Save images as files
    for i, (image, label) in enumerate(dataset):
        image = transforms.ToPILImage()(image)  # Convert to PIL Image
        image.save(os.path.join(SAVE_FOLDER, f"image_{i}.png"))

    print(f"Downloaded and saved {len(dataset)} images in {SAVE_FOLDER} at {IMG_SIZE} resolution.")

if __name__ == "__main__":
    main()
