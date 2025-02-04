import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, Subset
import random
from PIL import Image

# Configuration options
DATASET_NAME = "STL10"  # Options: "STL10", "CelebA", "ImageNet"
SAVE_FOLDER = "./data/images"  # Folder to save images
NUM_IMAGES = 100  # Set number of images (None for all)
IMG_SIZE = (512, 512)  # Desired resolution

# Create save folder if not exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Load dataset based on user selection
def get_dataset(name):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),  # Resize to 512x512
        transforms.ToTensor()
    ])

    if name == "STL10":
        return datasets.STL10(root=SAVE_FOLDER, download=True, split='train', transform=transform)
    elif name == "CelebA":
        return datasets.CelebA(root=SAVE_FOLDER, download=True, split='train', transform=transform)
    elif name == "ImageNet":
        return datasets.ImageFolder(root="/path/to/imagenet", transform=transform)  # Needs manual download
    else:
        raise ValueError("Unsupported dataset. Choose from STL10, CelebA, or ImageNet.")

# Load the dataset
dataset = get_dataset(DATASET_NAME)

# Select a subset of images
if NUM_IMAGES:
    indices = random.sample(range(len(dataset)), min(NUM_IMAGES, len(dataset)))
    dataset = Subset(dataset, indices)

# Save images as files
for i, (image, label) in enumerate(dataset):
    image = transforms.ToPILImage()(image)  # Convert to PIL Image
    image.save(os.path.join(SAVE_FOLDER, f"image_{i}.png"))

print(f"Downloaded and saved {len(dataset)} images in {SAVE_FOLDER} at {IMG_SIZE} resolution.")
