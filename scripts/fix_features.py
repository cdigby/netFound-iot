import torch
import joblib
import os

hr_dir = "/mnt/extra/models/iot2023-hr-http-unfreeze-ap"

train_features_path = os.path.join(hr_dir, "train_features.joblib")
train_labels_path = os.path.join(hr_dir, "train_labels.joblib")

test_features_path = os.path.join(hr_dir, "test_features.joblib.old")
test_labels_path = os.path.join(hr_dir, "test_labels.joblib.old")

# test_features_path = os.path.join(hr_dir, "test_features.joblib")
# test_labels_path = os.path.join(hr_dir, "test_labels.joblib")

train_features = joblib.load(train_features_path)
train_labels = joblib.load(train_labels_path)

test_features = joblib.load(test_features_path)
test_labels = joblib.load(test_labels_path)

print("train features")
print(len(train_features))
print(train_features)

print("test features")
print(len(test_features))
print(test_features)

print("train labels")
print(len(train_labels))
print(train_labels)

print("test labels")
print(len(test_labels))
print(test_labels)

new_test_features = test_features[len(train_features):].clone()
new_test_labels = test_labels[len(train_labels):].clone()

print("new test features")
print(len(new_test_features))
print(new_test_features)

print("new test labels")
print(len(new_test_labels))
print(new_test_labels)

print("sum of length of train and new test")
print(len(new_test_features) + len(train_features))
print(len(new_test_labels) + len(train_labels))

joblib.dump(new_test_features, os.path.join(hr_dir, "test_features.joblib"))
joblib.dump(new_test_labels, os.path.join(hr_dir, "test_labels.joblib"))
