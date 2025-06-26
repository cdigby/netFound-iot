from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import Trainer


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

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Performs a prediction step on the model and returns the loss, logits, and labels.
        This override bypasses compute_loss, which expects a loss from the model.
        """
        # Ensure model is in evaluation mode and inputs are on the correct device
        model.eval()
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            # Get model outputs directly
            outputs = model(**inputs)
        
        # The logits are the direct predictions from our Random Forest head
        logits = outputs.logits

        # Labels are in the inputs dictionary
        labels = inputs.get("labels")

        # By returning `None` for the loss, we tell the evaluation loop that
        # we don't have a loss value to report, which is fine.
        # The evaluation_loop will then skip trying to average the loss.
        loss = None
        
        return (loss, logits, labels)
