import pandas as pd

# Define the dataset directory and CSV file path
csv_file = "data/sample_submission.csv"

# Read the CSV file into a DataFrame
labels_df = pd.read_csv(csv_file)

# Clean and split the labels, converting them to lowercase
labels_df['label_name'] = labels_df['label_name'].str.split(';').apply(
    lambda x: [label.strip().lower() for label in x]
)

# Dictionary for correcting names
correction_dict = {
    'senne': 'senne',
    'youssef': 'youssef',
    'akif': 'akif',
    'seppe': 'seppe',
    'michiel': 'michiel',
    'raul': 'raul',
    'matthias': 'matthias',
    'florian': 'florian',
    'bart': 'bart',
    'nelli': 'nelli',
    'lasse': 'lasse',
    'alper': 'alper',
    'konrad': 'konrad',
    'daiane': 'daiane',
    'nothing': 'nothing',
    'alif': 'akif',  # Corrected spelling
    'asper': 'alper',
    'nille': 'nelli',
    'floarian': 'florian',
    'lesse': 'lasse',
    'alpre': 'alper',
    'mattias': 'matthias'
}

# Function to apply corrections
def correct_names(labels):
    return [correction_dict.get(label, label) for label in labels]

# Correct the names in the DataFrame
labels_df['label_name'] = labels_df['label_name'].apply(correct_names)

# Flatten the labels to count occurrences
label_counts = pd.Series([label for labels in labels_df['label_name'] for label in labels]).value_counts()

# Print the label counts
print(label_counts)

# Join the corrected labels back into a single string for saving
labels_df['label_name'] = labels_df['label_name'].apply('; '.join)

# Save the cleaned DataFrame back to a CSV file
output_file = "data/cleaned_labels.csv"
labels_df.to_csv(output_file, index=False)

print(f"Cleaned labels saved to {output_file}")
