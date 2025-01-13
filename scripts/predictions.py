import os
import numpy as np
from PIL import Image
import joblib
import pandas as pd
from keras_facenet import FaceNet

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------
CROPPED_FACES_DIR = './processed_data/cropped_faces_test'
CSV_OUTPUT_PATH = './results/catboost_predictions.csv'
MODEL_PATH = './classifiers/catboost/catboost_face_classifier.pkl'
NORMALIZER_PATH = './classifiers/catboost/catboost_normalizer.pkl'
LABEL_ENCODER_PATH = './classifiers/catboost/catboost_label_encoder.pkl'
IMG_SIZE = (160, 160)  # FaceNet expects images to be 160x160

# Exclude specific image number
EXCLUDED_IMAGE_NUM = '456'

# Load the saved model, normalizer, and label encoder
clf = joblib.load(MODEL_PATH)
in_encoder = joblib.load(NORMALIZER_PATH)
out_encoder = joblib.load(LABEL_ENCODER_PATH)

# Initialize the FaceNet embedder
embedder = FaceNet()

def predict_image(image_path):
    """
    Predicts the class of a single cropped face image.
    Args:
      image_path: Path to the image file.
    Returns:
      Prediction: (class_name, probability)
    """
    try:
        # Load and preprocess the image
        pil_img = Image.open(image_path).convert("RGB")
        pil_img = pil_img.resize(IMG_SIZE)
        img_array = np.array(pil_img, dtype="float32")
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 160, 160, 3)

        # Get the embedding
        embedding = embedder.embeddings(img_array)  # Shape: (1, d)

        # Normalize the embedding
        embedding_norm = in_encoder.transform(embedding)

        # Predict the class
        yhat_class = clf.predict(embedding_norm)[0]
        yhat_prob = clf.predict_proba(embedding_norm)[0, yhat_class]
        pred_name = out_encoder.inverse_transform([yhat_class])[0]

        return pred_name.lower(), yhat_prob  # Convert name to lowercase
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def process_cropped_faces(folder_path, csv_output_path):
    """
    Processes all cropped face images in a folder, orders them left-to-right based on filename,
    generates predictions, and saves them into a CSV file with one line per image.
    Args:
      folder_path: Path to the folder containing cropped face images.
      csv_output_path: Path to save the output CSV file.
    """
    predictions = {}

    # Iterate through all images in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)

            # Handle 'nothing' files
            if '_nothing' in filename:
                image_num = filename.split('_')[0].lstrip('0')  # Remove leading zeros
                predictions[image_num] = ['nothing']
                continue

            # Extract the image number and face number from the filename
            parts = filename.split('_')
            if len(parts) < 3:
                print(f"Skipping unexpected filename format: {filename}")
                continue

            image_num = parts[0].lstrip('0')  # Remove leading zeros

            # Skip the excluded image number
            if image_num == EXCLUDED_IMAGE_NUM:
                print(f"Skipping excluded image {EXCLUDED_IMAGE_NUM}")
                continue

            # Predict the label for each face
            pred_name, _ = predict_image(image_path)

            face_num = int(parts[2].split('.')[0])  # Extract the face number

            if pred_name:
                if image_num not in predictions:
                    predictions[image_num] = []
                predictions[image_num].append((face_num, pred_name))

    # Combine predictions for the same image number, ordered by face number
    final_predictions = []
    for image_num, names in predictions.items():
        if names == ['nothing']:  # If the file indicates 'nothing'
            label_name = 'nothing'
        else:
            # Sort faces by face number and combine labels
            label_name = ";".join([name for _, name in sorted(names)])
        final_predictions.append({"image": image_num, "label_name": label_name})

    # Save to CSV
    predictions_df = pd.DataFrame(final_predictions)
    predictions_df.to_csv(csv_output_path, index=False)
    print(f"Predictions saved to {csv_output_path}")

# Main Function
if __name__ == "__main__":
    process_cropped_faces(CROPPED_FACES_DIR, CSV_OUTPUT_PATH)
