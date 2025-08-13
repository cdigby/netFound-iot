import joblib
import numpy as np
import torch # You may also need torch as the joblib file contains a torch.Tensor

# joblib file contains a torch.Tensor, so need to call .detach().cpu().numpy() before it is useful
features = joblib.load("/mnt/extra/models/iot2023-hr-6layer/train_features.joblib").detach().cpu().numpy()
posioned_features = joblib.load("/mnt/extra/models/iot2023-hr-6layer-3m-hammered/train_features.joblib").detach().cpu().numpy()
print(type(features))

# features_bytes = features.tobytes() 



# # # NOW POISON FEATURES_BYTES
# # poisoned_bytes = bytearray()
# # for i in range(len(features_bytes)):
# #   poisoned_bytes.extend(features_bytes[i].to_bytes())

# # poisoned_features = np.frombuffer(poisoned_bytes, dtype=features.dtype).reshape(features.shape)

# # # Now save poisoned_features...

print(features.shape)
print(features[0])
print(features[1])
print(features[2])

print(posioned_features.shape)
print(posioned_features[0])
print(posioned_features[1])
print(posioned_features[2])

print(np.array_equal(features, posioned_features))
differences = np.not_equal(features, posioned_features)
print(differences.nonzero())

# for i, sample in enumerate(features):
#   sample_bytes = sample.tobytes()

#   # POISON this sample
#   sample_bytearray = bytearray(sample_bytes)
#   if i == 1:
#     sample_bytearray[3] = 36

#   features[i] = np.frombuffer(sample_bytearray, dtype=sample.dtype)


# print(features.shape)
# print(features[0])
# print(features[1])
# print(features[2])
