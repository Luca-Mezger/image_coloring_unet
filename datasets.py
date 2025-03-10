import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
from skimage import color  # Added for Lab conversion

def load_dataset(batch_size=32, img_size=256, data_dir='data/images'):
    """
    Load images and create (L channel, ab channels) pairs in Lab color space.
    Only files with .jpg, .jpeg, or .png extensions are processed.
    
    Args:
        batch_size (int): Number of images per batch.
        img_size (int): Target height and width of images.
        data_dir (str): Directory containing image files.
    
    Returns:
        tf.data.Dataset: Yields batches of (L, ab) image pairs.
    """
    print(f"Loading images from {data_dir}...")
    valid_extensions = ('.jpg', '.jpeg', '.png')
    file_list = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(data_dir, f))
    ]
    
    num_files = len(file_list)
    print(f"Found {num_files} images in {data_dir}")

    def gen():
        for file_path in tqdm(file_list, desc="Processing images", unit="image"):
            try:
                img = tf.io.read_file(file_path)
                img = tf.image.decode_jpeg(img, channels=3)
            except tf.errors.InvalidArgumentError:
                print(f"Skipping file (not a valid image): {file_path}")
                continue

            # Resize and convert to float32 [0, 1]
            img = tf.image.resize(img, [img_size, img_size])
            img = tf.cast(img, tf.float32) / 255.0

            # Convert to Lab color space
            lab = color.rgb2lab(img.numpy()).astype(np.float32)  # Convert to numpy for skimage
            
            # Extract and normalize channels
            L = (lab[..., 0] / 50.0) - 1.0  # Normalize L to [-1, 1]
            ab = lab[..., 1:] / 128.0        # Normalize ab to [-1, 1]

            # Add channel dimension to L
            L = tf.expand_dims(L, axis=-1)
            
            yield L, ab

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(img_size, img_size, 2), dtype=tf.float32)
        )
    )
    dataset = dataset.shuffle(buffer_size=num_files)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset