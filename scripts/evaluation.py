import pandas as pd

# Load the actual labels
actual_labels = pd.read_csv('./data/testset_labels.csv')

# Load the predictions
predicted_labels = pd.read_csv('./results/catboost_predictions_best.csv')

# Ensure both files have the same image IDs
if not actual_labels['image'].equals(predicted_labels['image']):
    raise ValueError("Image IDs in actual and predicted CSVs do not match.")

# Initialize metrics
total_images = len(actual_labels)
hamming_losses = []
accuracies = []
all_precision = []
all_recall = []
all_f1 = []

# For tracking mistakes (false positives, false negatives)
mistakes_data = []

# Evaluate per image
for i in range(total_images):
    image_id = actual_labels.iloc[i]['image']
    
    # ----------------------------------------------------------------
    # 1) Split into lists for ORDERED ACCURACY
    # ----------------------------------------------------------------
    actual_list = actual_labels.iloc[i]['label_name'].split(';')
    predicted_list = predicted_labels.iloc[i]['label_name'].split(';')

    # ----------------------------------------------------------------
    # 2) Also convert them to sets for the other metrics
    # ----------------------------------------------------------------
    actual_set = set(actual_list)
    predicted_set = set(predicted_list)
    
    # ----------------------------------------------------------------
    # HAMMING LOSS (set-based)
    # ----------------------------------------------------------------
    union = actual_set.union(predicted_set)
    intersection = actual_set.intersection(predicted_set)
    if len(union) == 0:
        # If both are empty, no difference
        hamming_loss_image = 0
    else:
        hamming_loss_image = len(union - intersection) / len(union)
    hamming_losses.append(hamming_loss_image)
    
    # ----------------------------------------------------------------
    # ACCURACY (ordered comparison)
    #
    #   => 1 if the lists match exactly (same length, same order)
    #   => 0 otherwise
    # ----------------------------------------------------------------
    accuracy_image = 1 if actual_list == predicted_list else 0
    accuracies.append(accuracy_image)
    
    # ----------------------------------------------------------------
    # PRECISION, RECALL, F1 (set-based)
    # ----------------------------------------------------------------
    tp = len(intersection)
    fp = len(predicted_set - actual_set)
    fn = len(actual_set - predicted_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    all_precision.append(precision)
    all_recall.append(recall)
    all_f1.append(f1)
    
    # ----------------------------------------------------------------
    # Track mistakes
    # ----------------------------------------------------------------
    false_positives = predicted_set - actual_set
    false_negatives = actual_set - predicted_set
    
    mistakes_data.append({
        "image": image_id,
        "actual_labels": ";".join(sorted(actual_set)),
        "predicted_labels": ";".join(sorted(predicted_set)),
        "false_positives": ";".join(sorted(false_positives)) if false_positives else "",
        "false_negatives": ";".join(sorted(false_negatives)) if false_negatives else ""
    })

# ----------------------------------------------------------------
# Aggregate metrics
# ----------------------------------------------------------------
overall_hamming = sum(hamming_losses) / total_images
overall_accuracy = sum(accuracies) / total_images
avg_precision = sum(all_precision) / total_images
avg_recall    = sum(all_recall) / total_images
avg_f1        = sum(all_f1) / total_images

print(f"Overall Hamming Loss: {overall_hamming:.4f}")
print(f"Overall Accuracy (order-sensitive): {overall_accuracy:.4f}")
print(f"Average Precision (set-based): {avg_precision:.4f}")
print(f"Average Recall (set-based): {avg_recall:.4f}")
print(f"Average F1-Score (set-based): {avg_f1:.4f}")

# ----------------------------------------------------------------
# Show only the mistakes
# ----------------------------------------------------------------
mistakes_df = pd.DataFrame(mistakes_data, columns=[
    "image",
    "actual_labels",
    "predicted_labels",
    "false_positives",
    "false_negatives"
])

# Filter out rows that have no false positives and no false negatives
bad_mask = (mistakes_df["false_positives"] != "") | (mistakes_df["false_negatives"] != "")
bad_mistakes_df = mistakes_df[bad_mask]

print("\n=== Mistakes per Image (Only Bad Predictions) ===")
if not bad_mistakes_df.empty:
    print(bad_mistakes_df.to_string(index=False))
else:
    print("No mistakes found!")

# Optional: Save mistakes to CSV
# bad_mistakes_df.to_csv('./results2/mistakes.csv', index=False)
