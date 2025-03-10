import argparse
import os
from PIL import Image
import numpy as np
from skimage import color

def increase_saturation(input_dir, output_dir, saturation_factor):
    """
    Reads all images from input_dir, converts them to Lab color space,
    scales the a and b channels by saturation_factor to enhance saturation,
    and saves the resulting RGB image to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_exts):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with Image.open(input_path) as img:
                img = img.convert('RGB')
                img_np = np.array(img) / 255.0  # Normalize to [0, 1]
                
                # Convert to Lab color space
                lab = color.rgb2lab(img_np)
                
                # Enhance saturation: scale a and b channels
                lab[..., 1] *= saturation_factor
                lab[..., 2] *= saturation_factor
                
                # Clip to valid Lab ranges
                lab[..., 1] = np.clip(lab[..., 1], -128, 127)
                lab[..., 2] = np.clip(lab[..., 2], -128, 127)
                
                # Convert back to RGB and scale to [0, 255]
                rgb_enhanced = color.lab2rgb(lab)
                rgb_enhanced = (rgb_enhanced * 255).astype(np.uint8)
                enhanced_img = Image.fromarray(rgb_enhanced)
                enhanced_img.save(output_path)
                
            print(f"Processed '{filename}' -> '{output_path}'")
        else:
            print(f"Skipping '{filename}' (not an image).")

def main():
    parser = argparse.ArgumentParser(
        description="Enhance image saturation in Lab color space."
    )
    parser.add_argument("--input_dir", required=True, type=str,
                        help="Path to the input folder containing images.")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Path to the output folder where processed images will be saved.")
    parser.add_argument("--saturation", default=1.5, type=float,
                        help="Factor by which to enhance saturation (1.0 = no change).")
    
    args = parser.parse_args()
    increase_saturation(args.input_dir, args.output_dir, args.saturation)

if __name__ == "__main__":
    main()
