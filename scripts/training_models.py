import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from PIL import Image
import glob
import joblib

from keras_facenet import FaceNet  # FaceNet from keras-facenet package

# 1) Initialize the FaceNet embedder
embedder = FaceNet()

# -------------------------------------------------------------------
# 2) Parameters
# -------------------------------------------------------------------
CROPPED_DIR = "./processed_data/cropped_faces"        # Path to your cropped_faces folder
EXCLUDE_FOLDERS = {"Nothing", "unmatched_data"}
IMG_SIZE = (160, 160)                  # FaceNet expects 160x160 by default

# -------------------------------------------------------------------
# 3) Helper: load all images from subfolders
# -------------------------------------------------------------------
def load_cropped_faces(cropped_dir=CROPPED_DIR, exclude=EXCLUDE_FOLDERS):
    X_images = []
    y_labels = []
    for class_name in os.listdir(cropped_dir):
        if class_name in exclude:
            print(f"Skipping excluded folder: {class_name}")
            continue
        subdir_path = os.path.join(cropped_dir, class_name)
        if not os.path.isdir(subdir_path):
            continue
        image_files = glob.glob(os.path.join(subdir_path, "*.*"))
        for img_path in image_files:
            try:
                pil_img = Image.open(img_path).convert("RGB")
                pil_img = pil_img.resize(IMG_SIZE)
                img_array = np.array(pil_img)
                X_images.append(img_array)
                y_labels.append(class_name)
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
        print(f"Loaded {len(image_files)} images for class={class_name}")
    return np.array(X_images), np.array(y_labels)

# -------------------------------------------------------------------
# 4) Create embeddings for each image
# -------------------------------------------------------------------
def get_embeddings(images):
    embeddings = embedder.embeddings(images)
    return embeddings

# -------------------------------------------------------------------
# 5) Main Code
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 5.1 Load all images (cropped faces) into memory
    X_images, y_labels = load_cropped_faces()
    print("Total images:", X_images.shape, "labels:", y_labels.shape)

    # 5.2 Create embeddings
    X_images = X_images.astype("float32")
    X_emb = get_embeddings(X_images)
    print("Embeddings shape:", X_emb.shape)

    # 5.3 Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_emb, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # 5.4 Normalize embeddings L2
    in_encoder = Normalizer(norm='l2')
    X_train_norm = in_encoder.transform(X_train)
    X_test_norm  = in_encoder.transform(X_test)

    # 5.5 Label encode
    out_encoder = LabelEncoder()
    out_encoder.fit(y_train)
    y_train_enc = out_encoder.transform(y_train)
    y_test_enc  = out_encoder.transform(y_test)

    # 5.6 Train SVM classifier
    #clf = SVC(kernel='linear', probability=True)
    #clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    #clf = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
    #clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42)
    clf = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=10, random_state=42, verbose=0)
    #clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')  # You can tune `n_neighbors`
    #clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    
    clf.fit(X_train_norm, y_train_enc)

    #clf.fit(X_train_norm, y_train_enc)

    # 5.7 Evaluate
    train_acc = clf.score(X_train_norm, y_train_enc)
    test_acc  = clf.score(X_test_norm,  y_test_enc)
    print(f"Accuracy: train={train_acc*100:.2f}%, test={test_acc*100:.2f}%")

    # 5.8 Save the model, label encoder, and normalizer
    joblib.dump(clf, './classifiers/catboost/catboost_face_classifier.pkl')
    joblib.dump(in_encoder, './classifiers/catboost/catboost_normalizer.pkl')
    joblib.dump(out_encoder, './classifiers/catboost/catboost_label_encoder.pkl')
    print("Model, normalizer, and label encoder saved successfully!")

    # 5.9 Example prediction on one test sample
    from random import choice
    idx = choice(range(len(X_test_norm)))
    sample_emb = np.expand_dims(X_test_norm[idx], axis=0)
    yhat_class = clf.predict(sample_emb)[0]
    yhat_prob  = clf.predict_proba(sample_emb)[0, yhat_class]
    pred_name  = out_encoder.inverse_transform([yhat_class])[0]

    print("\n--- Example Prediction ---")
    print("Ground truth:", y_test[idx])
    print(f"Predicted: {pred_name} ({yhat_prob*100:.2f}%)")
