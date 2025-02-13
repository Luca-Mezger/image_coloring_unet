import argparse
import os
from PIL import Image, ImageEnhance

def increase_saturation(input_dir, output_dir, saturation_factor):
    """
    Reads all images from input_dir, enhances their color saturation,
    and saves them to output_dir.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define valid image extensions (add more if you like)
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    
    # Iterate over all files in input_dir
    for filename in os.listdir(input_dir):
        # Check extension
        if filename.lower().endswith(valid_exts):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Open the image
            with Image.open(input_path) as img:
                # Enhance color (this affects saturation)
                enhancer = ImageEnhance.Color(img)
                enhanced_img = enhancer.enhance(saturation_factor)
                
                # Save to output directory
                enhanced_img.save(output_path)
                
            print(f"Processed '{filename}' -> '{output_path}'")
        else:
            print(f"Skipping '{filename}' (not an image).")

def main():
    parser = argparse.ArgumentParser(
        description="Increase saturation of all images in a folder and copy them to a new folder."
    )
    parser.add_argument(
        "--input_dir", 
        required=True, 
        type=str, 
        help="Path to the input folder containing images."
    )
    parser.add_argument(
        "--output_dir", 
        required=True, 
        type=str, 
        help="Path to the output folder where processed images will be saved."
    )
    parser.add_argument(
        "--saturation", 
        default=1.5, 
        type=float, 
        help="Factor by which to enhance color saturation. (1.0 = no change)"
    )

    args = parser.parse_args()
    
    # Run the saturation enhancement
    increase_saturation(args.input_dir, args.output_dir, args.saturation)

if __name__ == "__main__":
    main()
