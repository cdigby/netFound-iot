import numpy as np
import pandas as pd
import joblib
import os
import shutil

HR_DIR = "/mnt/extra/models/iot2023-hr-ft-base"
OUTPUT_DIR = "/mnt/extra/models/iot2023-hr-gaussian-injection"

MINORITY_CLASSES = [4, 5, 7]

NOISE_LEVEL = 0.01

print(f"load hidden representation of dataset from {HR_DIR}")
train_features_path = os.path.join(HR_DIR, "train_features.joblib")
train_labels_path = os.path.join(HR_DIR, "train_labels.joblib")

train_features = joblib.load(train_features_path).detach().cpu().numpy()
train_labels = joblib.load(train_labels_path).detach().cpu().numpy()

new_samples = []
new_labels = []

feature_std_devs = np.std(train_features, axis=0)

minority_indices = np.nonzero(np.isin(train_labels, MINORITY_CLASSES))[0]
minority_features = train_features[minority_indices]

print("Original class distribution")
print(pd.Series(train_labels).value_counts().sort_index())

# Generate new samples by applying gaussian noise to minority samples
# We do this on the feature vectors extracted by NetfoundFeatureExtractor
print("generate new samples")
for i, sample in enumerate(minority_features):
  noise = np.random.normal(
    loc=0.0, 
    scale=feature_std_devs * NOISE_LEVEL, 
    size=sample.shape
  )
  
  new_samples.append(sample + noise)
  new_labels.append(train_labels[minority_indices[i]])

# Append new samples
print(f"add {len(new_samples)} new samples to train dataset")
train_features = np.concatenate((train_features, new_samples), axis=0)
train_labels = np.concatenate((train_labels, new_labels), axis=0)

# Shuffle both arrays in unison
shuffle_indices = np.random.permutation(len(train_features))
train_features = train_features[shuffle_indices]
train_labels = train_labels[shuffle_indices]

print("New class distribution")
print(pd.Series(train_labels).value_counts().sort_index())

# Save
print(f"save augmented hidden representation to {OUTPUT_DIR}")
if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)
joblib.dump(train_features, os.path.join(OUTPUT_DIR, "train_features.joblib"))
joblib.dump(train_labels, os.path.join(OUTPUT_DIR, "train_labels.joblib"))

# Copy original test set to output directory
print(f"copy original test set to {OUTPUT_DIR}")
test_features_path = os.path.join(HR_DIR, "test_features.joblib")
test_labels_path = os.path.join(HR_DIR, "test_labels.joblib")

shutil.copyfile(test_features_path, os.path.join(OUTPUT_DIR, "test_features.joblib"))
shutil.copyfile(test_labels_path, os.path.join(OUTPUT_DIR, "test_labels.joblib"))