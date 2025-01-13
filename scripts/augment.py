import os
import glob
import numpy as np
from PIL import Image

try:
    # If you're on TF 2.x, import from tensorflow.keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    # Otherwise, fallback to the older keras package
    from keras.preprocessing.image import ImageDataGenerator


##############################################################################
# Global Config (Adjust as needed)
##############################################################################
IMG_SIZE = (160, 160)  # FaceNet expects 160x160

# (Optional) small helper to check if a file is likely an image.
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def is_image_file(filepath):
    _, ext = os.path.splitext(filepath.lower())
    return ext in VALID_EXTENSIONS


def augment_images(
    input_dir,
    output_dir,
    augment_per_image=5,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    seed=None,  # If you want reproducible results, pass a fixed seed.
):
    """
    Augments images found in `input_dir` (organized by class) and saves them
    to `output_dir`, preserving the same subfolder structure.

    Args:
        input_dir (str): Directory containing subfolders of images (one folder per class).
        output_dir (str): Directory where augmented images will be saved.
        augment_per_image (int): Number of augmented samples to generate per original image.
        rotation_range, width_shift_range, ...: Parameters for ImageDataGenerator.
        seed (int): A random seed for reproducible augmentations (applied in .flow()).
    """
    # Safety check: avoid writing to the exact same directory as input
    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        raise ValueError(
            "Input directory and output directory must be different to avoid overwriting."
        )

    # Initialize the data generator (no 'seed' argument here)
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode,
    )

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Counters for summary
    total_original = 0
    total_augmented = 0

    # Walk through each class folder
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue  # Skip files that aren't directories

        # Create the same class subfolder in the output directory
        out_class_path = os.path.join(output_dir, class_name)
        os.makedirs(out_class_path, exist_ok=True)

        # Process each image in the current class folder
        for img_file in glob.glob(os.path.join(class_path, "*.*")):
            if not is_image_file(img_file):
                continue  # Skip non-image files

            try:
                # Load and resize image
                img = Image.open(img_file).convert("RGB")
                img = img.resize(IMG_SIZE)
                img_array = np.array(img)
                # Expand dims to create a batch of size 1
                img_array = np.expand_dims(img_array, axis=0)

                # Augment images and save to disk
                aug_count = 0
                # Pass the seed to .flow() if provided
                gen = datagen.flow(img_array, batch_size=1, seed=seed)

                for batch in gen:
                    aug_img = Image.fromarray(batch[0].astype("uint8"))
                    out_filename = f"aug_{aug_count}_{os.path.basename(img_file)}"
                    aug_img.save(os.path.join(out_class_path, out_filename))

                    aug_count += 1
                    if aug_count >= augment_per_image:
                        break

                total_original += 1
                total_augmented += aug_count

            except Exception as e:
                print(f"[Warning] Error processing {img_file}: {e}")

    print("=== Data Augmentation Summary ===")
    print(f"Total original images found: {total_original}")
    print(f"Total augmented images saved: {total_augmented}")
    print(f"Augmented images saved in: {output_dir}")


##############################################################################
# Example usage
##############################################################################
if __name__ == "__main__":
    original_data_dir = "./processed_data/cropped_faces"  
    augmented_data_dir = "./processed_data/cropped_faces_augmented"

    augment_images(
        input_dir=original_data_dir,
        output_dir=augmented_data_dir,
        augment_per_image=5,         # number of augmented samples per image
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        seed=42,  # Remove or change this if you don't need reproducible transformations
    )

    print("Data augmentation completed.")
