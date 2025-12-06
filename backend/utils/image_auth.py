# identify corrupted files within datasets
# initial code from https://drlee.io/identifying-corrupted-images-before-feeding-them-into-a-cnn-13397844ef3c
import os
from PIL import Image
import tensorflow as tf
import warnings
warnings.filterwarnings("error")

def validate_images(directory):
    corrupted_files = []
    
    # Walk through directory and sub-directories
    for dirpath, _, filenames in os.walk(directory):
        print(f"Scanning directory: {dirpath}")
        
        for image_file in filenames:
            # Check for common image extensions
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                image_path = os.path.join(dirpath, image_file)
                try:
                    #use tensorflow to check if it can process the images
                    image_data = tf.io.read_file(image_path)
                    _ = tf.image.decode_image(image_data)
                except Exception as e:
                    # describe which images get errors and why
                    corrupted_files.append(image_path)
                    print(f"Error with {image_path}: {e}")
    
    return corrupted_files

directory = "<your/file/path>"  # Make sure to change to YOUR directory!
corrupted_images = validate_images(directory)

# confirmation messages
if corrupted_images:
    print(f"Found {len(corrupted_images)} corrupted images.")
else:
    print("All images are valid!")