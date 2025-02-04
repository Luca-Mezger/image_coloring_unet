import tensorflow as tf
import os

def load_dataset(batch_size=32, img_size=256, data_dir='data/images'):
    """
    Load images from a directory, resize them, and create (grayscale, color) pairs.
    
    Args:
        batch_size (int): Number of images per batch.
        img_size (int): Target height and width of images.
        data_dir (str): Directory containing image files (JPEG/PNG).
    
    Returns:
        tf.data.Dataset: Yields batches of (grayscale, color) image pairs.
    """
    # Create a dataset of file paths from the data directory.
    file_pattern = os.path.join(data_dir, '*')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)

    def _process_image(file_path):
        # Read and decode the image.
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)  # Use decode_png if needed.
        # Resize and normalize the image.
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0
        # Convert the color image to grayscale.
        grayscale = tf.image.rgb_to_grayscale(img)
        return grayscale, img

    # Map the processing function, shuffle, batch, and prefetch for performance.
    dataset = dataset.map(_process_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
