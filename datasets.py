import tensorflow as tf
import os
from tqdm import tqdm

def load_dataset(batch_size=32, img_size=256, data_dir='data/images'):
    """
    Load images from a directory, resize them, and create (grayscale, color) pairs.
    Only files with .jpg, .jpeg, or .png extensions are processed.
    
    Args:
        batch_size (int): Number of images per batch.
        img_size (int): Target height and width of images.
        data_dir (str): Directory containing image files.
    
    Returns:
        tf.data.Dataset: Yields batches of (grayscale, color) image pairs.
    """
    # Filter for valid image files
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
                img = tf.image.decode_jpeg(img, channels=3)  # Use decode_png if needed.
            except tf.errors.InvalidArgumentError:
                print(f"Skipping file (not a valid image): {file_path}")
                continue
            # Resize and normalize the image.
            img = tf.image.resize(img, [img_size, img_size])
            img = tf.cast(img, tf.float32) / 255.0
            # Convert the color image to grayscale.
            grayscale = tf.image.rgb_to_grayscale(img)
            yield grayscale, img

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32)
        )
    )
    dataset = dataset.shuffle(buffer_size=num_files)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
