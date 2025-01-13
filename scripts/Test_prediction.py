import numpy as np
from PIL import Image
import joblib
from keras_facenet import FaceNet

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------
MODEL_PATH = './classifiers/catboost/catboost_face_classifier.pkl'
NORMALIZER_PATH = './classifiers/catboost/catboost_normalizer.pkl'
LABEL_ENCODER_PATH = './classifiers/catboost/catboost_label_encoder.pkl'
IMG_SIZE = (160, 160)  # FaceNet expects images to be 160x160

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

        return pred_name, yhat_prob
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

# Test the function with a single image
image_path = "./processed_data/cropped_faces_test/0040_face_4.jpg"  # Replace with the actual path
predicted_class, confidence = predict_image(image_path)

if predicted_class:
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence*100:.2f}%")
else:
    print("Failed to make a prediction.")
