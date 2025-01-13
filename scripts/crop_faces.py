import os
import cv2  # OpenCV for YuNet model and video processing
from skimage.transform import resize
from PIL import Image, ExifTags
import onnx
import pillow_heif  # Enables HEIC/HEIF support in Pillow
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
dataset_dir = "./data/trainset"
csv_file = "./data/cleaned_labels.csv"
output_dir = "./processed_data/cropped_faces"
model_path = "./models/face_detection_yunet_2023mar.onnx"

# Enable HEIF support in Pillow
pillow_heif.register_heif_opener()

# Load YuNet model
face_detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(640, 640),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=5000
)

# Utility to handle EXIF orientation
def handle_exif_orientation(image):
    """Corrects the image orientation based on EXIF data."""
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, None)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"Warning: Unable to handle EXIF orientation due to {e}")
    return image

# Load and preprocess image
def preprocess_image(file_path, scale_factor=0.4):
    """Loads and preprocesses the image."""
    try:
        image = Image.open(file_path)
        image = handle_exif_orientation(image)
        image = np.array(image)

        # Downscale for faster processing
        resized_image = resize(
            image, 
            (int(image.shape[0] * scale_factor), int(image.shape[1] * scale_factor))
        )
        resized_image = (resized_image * 255).astype(np.uint8)
        return resized_image
    except Exception as e:
        print(f"Error preprocessing file {file_path}: {e}")
        return None

# Handle MP4 files to extract the first frame
def extract_first_frame_from_video(file_path):
    """Extract the first frame from a video file."""
    try:
        # Open the video file
        video_capture = cv2.VideoCapture(file_path)
        ret, frame = video_capture.read()
        if not ret:
            print(f"Error reading video {file_path}")
            return None

        # Convert the frame from BGR (OpenCV format) to RGB (PIL format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = Image.fromarray(frame_rgb)
        frame_rgb = np.array(frame_rgb)

        # Downscale for faster processing
        resized_frame = resize(
            frame_rgb, 
            (int(frame_rgb.shape[0] * 0.3), int(frame_rgb.shape[1] * 0.3))
        )
        resized_frame = (resized_frame * 255).astype(np.uint8)
        return resized_frame
    except Exception as e:
        print(f"Error extracting frame from video {file_path}: {e}")
        return None

# Detect and save faces
def detect_and_save_faces(image_array, detected_faces, output_dir, image_num, names, target_size=(160, 160)):
    """Detects, resizes, and saves cropped faces into corresponding label folders."""
    try:
        for i, ((x, y, w, h), name) in enumerate(zip(detected_faces, names)):
            # Clean up the name (strip spaces, capitalize first letter of each word)
            clean_name = name.strip().title()

            # Extract face region
            x, y, w, h = map(int, [x, y, w, h])
            cropped_face = image_array[y:y + h, x:x + w]

            # Resize to target size
            face_image = Image.fromarray(cropped_face).resize(target_size, Image.LANCZOS)

            # Ensure the label folder exists (name from sample_submission)
            label_folder = os.path.join(output_dir, clean_name)

            # Only create the folder if it doesn't exist yet
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)
                print(f"Created folder: {label_folder}")
            
            # Save the resized face
            face_path = os.path.join(label_folder, f"{image_num}_face_{i + 1}.jpg")
            face_image.save(face_path)
            print(f"Saved resized face {i + 1} to {face_path}")
    except Exception as e:
        print(f"Error saving cropped faces for image {image_num}: {e}")



# Detect faces using YuNet
def detect_faces_based_on_names(image_array, names):
    """Detect faces using the YuNet model and sort them left-to-right."""
    try:
        # Convert to grayscale if necessary
        if len(image_array.shape) == 3:
            grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            grayscale_image = image_array

        face_detector.setInputSize((grayscale_image.shape[1], grayscale_image.shape[0]))
        retval, faces = face_detector.detect(grayscale_image)

        if retval > 0:
            detected_faces = faces[:, :4]  # x, y, w, h
            # Sort faces left-to-right by the x-coordinate
            detected_faces = sorted(detected_faces, key=lambda face: face[0])  # Sort by x-coordinate
        else:
            detected_faces = []

        if len(detected_faces) == len(names):
            print(f"Detected {len(detected_faces)} faces, which matches the number of names ({len(names)}).")
        else:
            print(f"Warning: Detected {len(detected_faces)} faces, but there are {len(names)} names.")

        return detected_faces
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return []


# Save original image when faces are not matched
def save_original_image(image_array, image_num, names):
    """Saves the original image in a separate folder when faces are not detected correctly."""
    try:
        # Create a directory for saving unmatched data
        unmatched_dir = os.path.join(output_dir, "unmatched_data")
        os.makedirs(unmatched_dir, exist_ok=True)

        # Save the original image with the expected names in the filename
        original_image_path = os.path.join(
            unmatched_dir, f"{image_num}_unmatched_{'_'.join(names)}.jpg"
        )

        # Convert the image array to a PIL Image and save
        image = Image.fromarray(image_array)
        image.save(original_image_path)
        print(f"Saved unmatched image to {original_image_path}")
    except Exception as e:
        print(f"Error saving unmatched image {image_num}: {e}")

# Process a single image or video
def process_image(image_num, names):
    """Processes a single image or video by detecting faces and saving results."""
    file_path = next(
        (os.path.join(dataset_dir, f"{image_num}.{ext}")
         for ext in ["jpg", "jpeg", "png", "heic", "mp4"]
         if os.path.exists(os.path.join(dataset_dir, f"{image_num}.{ext}"))),
        None,
    )

    if not file_path:
        print(f"File {image_num} not found in any supported format. Skipping...")
        return

    # If the file is a video, extract the first frame
    if file_path.endswith(".mp4"):
        image_array = extract_first_frame_from_video(file_path)
    else:
        # Otherwise, treat it as an image
        image_array = preprocess_image(file_path)
    
    if image_array is None:
        return

    detected_faces = detect_faces_based_on_names(image_array, names)

    # If the number of detected faces matches the number of names, save the faces
    if len(detected_faces) == len(names):
        print(f"Image {image_num}: Detected {len(detected_faces)} faces, saving...")
        detect_and_save_faces(image_array, detected_faces, output_dir, image_num, names)
    else:
        print(f"Warning: Image {image_num} detected {len(detected_faces)} faces, but there are {len(names)} names.")
        
        # Save the original image (or frame) for inspection
        save_original_image(image_array, image_num, names)

# Main function for processing the dataset
def process_dataset():

    """Processes the entire dataset sequentially."""
    print("Parsing CSV file...")
    labels_df = pd.read_csv(csv_file)
    labels_df['image'] = labels_df['image'].apply(lambda x: f"{x:04d}")
    
    # Create a map of image to names
    image_label_map = dict(zip(labels_df['image'], labels_df['label_name']))

    print("Processing images and videos, saving cropped faces...")
    for image_num, names in image_label_map.items():
        names = names.split(';')  # Split by semicolon for multiple names in one image or video
        process_image(image_num, names)

# Run the pipeline
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    process_dataset()
