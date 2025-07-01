from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import Trainer

import joblib
import os

class NetfoundTrainer(Trainer):

    extraFields = {}

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        self._signature_columns += {
            "direction",
            "iats",
            "bytes",
            "pkt_count",
            "total_bursts",
            "ports",
            "stats",
            "protocol",
        }
        self._signature_columns += self.extraFields
        self._signature_columns = list(set(self._signature_columns))

    def __init__(self, label_names=None, extraFields = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if extraFields is not None:
            self.extraFields = extraFields
        if label_names is not None:
            self.label_names.extend(label_names)


class NetfoundFeatureExtractorTrainer(NetfoundTrainer):
    def __init__(self, label_names=None, extraFields = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This will store the extracted features from the NetFoundFeatureExtractor
        self.all_features = []
        self.all_labels = []

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        This prediction step has the sole purpose of extracting features so we can dump them to file
        Nothing is returned
        """
        # Ensure model is in evaluation mode and inputs are on the correct device
        model.eval()
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            # Get model outputs directly
            outputs = model(**inputs)

        # Labels are in the inputs dictionary
        labels = inputs.get("labels")

        self.all_features.append(outputs.features.cpu())
        self.all_labels.append(labels.cpu())
        
        return (None, None, None)
    
    def dump_features(self, output_dir, label):
        final_features = torch.cat(self.all_features, dim=0)
        final_labels = torch.cat(self.all_labels, dim=0)
        joblib.dump(final_features, os.path.join(output_dir, f"{label}_features.joblib"))
        joblib.dump(final_labels, os.path.join(output_dir, f"{label}_labels.joblib"))

    def reset_stored_features(self):
        self.all_features = []
        self.all_labels = []