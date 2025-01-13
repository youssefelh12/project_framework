import os
import cv2  # OpenCV for YuNet model and video processing
from skimage.transform import resize
from PIL import Image, ExifTags
import pillow_heif  # Enables HEIC/HEIF support in Pillow
import numpy as np
from pathlib import Path

# Paths
testset_dir = "./data/testset"              # Directory containing test images/videos
output_dir = "./processed_data/cropped_faces_test"
model_path = "./models/face_detection_yunet_2023mar.onnx"

# Enable HEIF support in Pillow
pillow_heif.register_heif_opener()

# Load YuNet model
face_detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(640, 640),
    score_threshold=0.73,
    nms_threshold=0.3,
    top_k=5000
)

# -------------------------------------------------------------------
# Utility to handle EXIF orientation
# -------------------------------------------------------------------
def handle_exif_orientation(image):
    """Corrects the image orientation based on EXIF data."""
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"Warning: Unable to handle EXIF orientation due to {e}")
    return image

# -------------------------------------------------------------------
# Load and preprocess image
# -------------------------------------------------------------------
def preprocess_image(file_path, scale_factor=0.4):
    """Loads and preprocesses the image or returns None if it fails."""
    try:
        pil_image = Image.open(file_path)
        pil_image = handle_exif_orientation(pil_image)
        np_image = np.array(pil_image)

        # Downscale for faster processing
        resized_image = resize(
            np_image, 
            (int(np_image.shape[0] * scale_factor), int(np_image.shape[1] * scale_factor))
        )
        resized_image = (resized_image * 255).astype(np.uint8)
        return resized_image
    except Exception as e:
        print(f"Error preprocessing file {file_path}: {e}")
        return None

# -------------------------------------------------------------------
# Handle MP4 files: extract the first frame
# -------------------------------------------------------------------
def extract_first_frame_from_video(file_path):
    """Extract the first frame from a video file, or None if it fails."""
    try:
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Error reading video {file_path}")
            return None

        # Convert the frame from BGR to RGB for consistency
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = Image.fromarray(frame_rgb)
        np_frame = np.array(frame_rgb)

        # Downscale for faster processing
        resized_frame = resize(
            np_frame, 
            (int(np_frame.shape[0] * 0.3), int(np_frame.shape[1] * 0.3))
        )
        resized_frame = (resized_frame * 255).astype(np.uint8)
        return resized_frame
    except Exception as e:
        print(f"Error extracting frame from video {file_path}: {e}")
        return None

# -------------------------------------------------------------------
# Detect faces using YuNet, returning bounding boxes and confidences
# -------------------------------------------------------------------
def detect_faces(image_array):
    """
    Runs YuNet detection on a given image_array (RGB -> BGR if needed).
    Returns a list of (box, confidence), where:
        box = [x, y, w, h]
        confidence = float
    Sorted left->right by x-coordinate of the bounding box.
    """
    try:
        if len(image_array.shape) == 3:
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = image_array

        face_detector.setInputSize((bgr_image.shape[1], bgr_image.shape[0]))
        retval, faces = face_detector.detect(bgr_image)

        if retval > 0 and faces is not None:
            # faces shape: (num_faces, 15)
            # columns: [x, y, w, h, landmark_1_x, ..., landmark_5_y, confidence]
            detections = []
            for face in faces:
                x, y, w, h = face[:4]
                conf = face[14]
                detections.append(([x, y, w, h], conf))

            # Sort left->right by bounding-box x
            detections = sorted(detections, key=lambda d: d[0][0])
            return detections
        else:
            return []
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return []

# -------------------------------------------------------------------
# Save one cropped face
# -------------------------------------------------------------------
def save_cropped_face(image_array, box, output_path, target_size=(160, 160), padding_ratio=0.2):
    """
    Adjusts bounding box to fit within image boundaries and adds padding 
    to ensure the full face is captured. Crops the image and saves it.
    """
    try:
        img_height, img_width = image_array.shape[:2]
        x, y, w, h = map(int, box)

        # Compute padding
        padding_x = int(w * padding_ratio)
        padding_y = int(h * padding_ratio)

        # Expand the bounding box
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(w + 2 * padding_x, img_width - x)
        h = min(h + 2 * padding_y, img_height - y)

        if w <= 0 or h <= 0:
            print(f"Adjusted bounding box is invalid: {box}. Saving entire image as fallback.")
            fallback_output_path = output_path.replace("_face_", "_fallback_")
            full_image = Image.fromarray(image_array)
            full_image.save(fallback_output_path)
            print(f"Saved fallback image to {fallback_output_path}")
            return

        # Crop the face region
        cropped_face = image_array[y:y+h, x:x+w]

        # Resize to target size
        face_pil = Image.fromarray(cropped_face).resize(target_size, Image.LANCZOS)
        face_pil.save(output_path)
        print(f"Saved face to {output_path}")

    except Exception as e:
        print(f"Error saving cropped face: {e}")

# -------------------------------------------------------------------
# If no faces, save entire image as *nothing.jpg
# -------------------------------------------------------------------
def save_nothing_image(image_array, output_path):
    """
    Saves the entire image when no face is detected.
    """
    try:
        pil_img = Image.fromarray(image_array)
        pil_img.save(output_path)
        print(f"No faces found, saved entire image as {output_path}")
    except Exception as e:
        print(f"Error saving 'nothing' image: {e}")

# -------------------------------------------------------------------
# Helper to rotate the image array by a given angle
# -------------------------------------------------------------------
def rotate_image(image_array, angle):
    """
    Rotates a given image array (in degrees) using PIL.
    """
    pil_img = Image.fromarray(image_array)
    rotated_pil = pil_img.rotate(angle, expand=True)
    return np.array(rotated_pil)

# -------------------------------------------------------------------
# Process a single image/video
# -------------------------------------------------------------------
def process_file(file_path):
    """
    1) Load the image or video first frame.
    2) For each orientation [0, 90, 180, 270]:
       - Detect faces
       - Sum detection confidences
    3) Pick the orientation with the highest total detection confidence
    4) If no orientation yields any faces, save entire image as _nothing.jpg
       Otherwise, only save faces (once) from that best orientation.
    """
    # Extract file name without extension for naming
    file_name = Path(file_path).stem

    # (1) Load image or MP4 first frame
    if file_path.lower().endswith(".mp4"):
        base_image = extract_first_frame_from_video(file_path)
    else:
        base_image = preprocess_image(file_path)

    if base_image is None:
        print(f"Failed to read {file_path}. Skipping.")
        return

    # Store each orientation's result
    candidates = []
    angles = [0, 90, 180, 270]

    for angle in angles:
        # Rotate the image only if angle != 0
        rotated_img = rotate_image(base_image, angle) if angle != 0 else base_image
        
        # Detect faces
        detections = detect_faces(rotated_img)
        
        # Sum the detection confidences
        total_conf = sum(det[1] for det in detections)

        candidates.append({
            'angle': angle,
            'rotated_img': rotated_img,
            'detections': detections,
            'total_conf': total_conf
        })

    # Pick the orientation with the highest total detection confidence
    best_candidate = max(candidates, key=lambda c: c['total_conf'])

    # If best orientation has zero confidence, it means no faces in any orientation
    if best_candidate['total_conf'] == 0:
        nothing_path = os.path.join(output_dir, f"{file_name}_nothing.jpg")
        save_nothing_image(base_image, nothing_path)
        return

    # Otherwise, save faces from that best orientation
    for i, (box, conf) in enumerate(best_candidate['detections'], start=1):
        face_path = os.path.join(output_dir, f"{file_name}_face_{i}.jpg")
        save_cropped_face(best_candidate['rotated_img'], box, face_path)

# -------------------------------------------------------------------
# Main function for testset
# -------------------------------------------------------------------
def process_testset():
    """
    Processes all images/videos in the testset directory.
    """
    print(f"Processing testset in {testset_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(testset_dir):
        file_path = os.path.join(testset_dir, file_name)
        # Filter out only valid image/video files
        if (
            os.path.isfile(file_path)
            and file_name.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".mp4"))
        ):
            process_file(file_path)

# -------------------------------------------------------------------
# Run the pipeline
# -------------------------------------------------------------------
if __name__ == "__main__":
    process_testset()
