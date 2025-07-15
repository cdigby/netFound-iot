import os, sys
import joblib
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

DATASET_DIR = "/mnt/extra/processed/iot2023/iot2023-8class-http"
HR_DIR = "/mnt/extra/models/iot2023-hr-ft-base"
TRAIN_OUTPUT_DIR = "/mnt/extra/processed/iot2023/iot2023-8class-crossover-train"
TEST_OUTPUT_DIR = "/mnt/extra/processed/iot2023/iot2023-8class-crossover-test"

TARGET_CLASSES = [4, 5, 7]

train_features_path = os.path.join(HR_DIR, "train_features.joblib")
train_labels_path = os.path.join(HR_DIR, "train_labels.joblib")

def crossover_augment(A, B):
  # Determine a crossover point (e.g., halfway through sample A)
  num_bursts_A = len(A['directions'])
  if num_bursts_A < 2:
    return None # Can't perform crossover on very short flows
  
  crossover_point = num_bursts_A // 2
  
  # Create the new sample by combining parts of A and B
  new_sample = {}
  
  # Take the first half of the lists from A and the second half from B
  new_sample['burst_tokens'] = A['burst_tokens'][:crossover_point] + B['burst_tokens'][crossover_point:]
  new_sample['directions'] = A['directions'][:crossover_point] + B['directions'][crossover_point:]
  new_sample['bytes'] = A['bytes'][:crossover_point] + B['bytes'][crossover_point:]
  new_sample['iats'] = A['iats'][:crossover_point] + B['iats'][crossover_point:]
  new_sample['counts'] = A['counts'][:crossover_point] + B['counts'][crossover_point:]
  
  # The scalar features can be taken from the primary parent, A
  new_sample['flow_duration'] = A['flow_duration']
  new_sample['protocol'] = A['protocol']
  new_sample['labels'] = A['labels']
  
  return new_sample

print("load hidden representation of dataset")
train_features = joblib.load(train_features_path).detach().cpu().numpy()
train_labels = joblib.load(train_labels_path).detach().cpu().numpy()

print("load original dataset")
train_dataset = load_dataset(
  "arrow",
  data_dir=DATASET_DIR,
  split=f"train[20%:]",
  streaming=False,
)

test_dataset = load_dataset(
  "arrow",
  data_dir=DATASET_DIR,
  split=f"train[:20%]",
  streaming=False,
)

print("fit nearest neighbors")
nn = NearestNeighbors(n_neighbors=5, algorithm="auto")
nn.fit(train_features)

print("find nearest neighbours of target samples")
target_indices = np.nonzero(np.isin(train_labels, TARGET_CLASSES))[0]
target_features = train_features[target_indices]
all_nearest = nn.kneighbors(target_features, return_distance=False)

print("crossover augment train dataset")
new_samples_list = []
for i, original_index in enumerate(tqdm(target_indices)):
  # Get neigbours of current sample
  nearest = all_nearest[i]

  # Find first neighbour that is the same class
  neighbour_index = None
  for neighbour_original_index in nearest:
    # Skip self
    if neighbour_original_index == original_index:
      continue

    if train_labels[neighbour_original_index] == train_labels[original_index]:
      neighbour_index = neighbour_original_index
      break

  if neighbour_index is None:
    # No neighbour found that is the same class
    continue

  # Crossover augment sample i with data from its neighbour
  new_sample = crossover_augment(train_dataset[int(original_index)], train_dataset[int(neighbour_index)])

  if new_sample is None:
    # Failed, probably too short
    continue

  # Add the new sample to the dataset
  new_samples_list.append(new_sample)

print(f"add {len(new_samples_list)} new samples to train dataset")
new_samples_dataset = Dataset.from_list(new_samples_list).cast(train_dataset.features)
train_dataset = concatenate_datasets([train_dataset, new_samples_dataset])

print("shuffle new train dataset")
train_dataset = train_dataset.shuffle(seed=42)

# Print frequency of each class in train_labels
print("Old class distribution in training set:")
print(pd.Series(train_labels).value_counts().sort_index())

# Print frequency of each class in train_dataset
print("Augmented class distribution in training set:")
print(pd.Series(train_dataset['labels']).value_counts().sort_index())

print("save train and test dataset")
train_dataset.save_to_disk(TRAIN_OUTPUT_DIR)
test_dataset.save_to_disk(TEST_OUTPUT_DIR)
