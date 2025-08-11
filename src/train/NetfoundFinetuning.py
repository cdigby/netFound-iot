import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch
import torch.distributed
import numpy as np
import utils
import random
import sys
import math
from dataclasses import field, dataclass
from datasets.distributed import split_dataset_by_node
from typing import Optional
from copy import deepcopy
from torchinfo import summary
from torch.distributed.elastic.multiprocessing.errors import record

from transformers import (
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    EarlyStoppingCallback,
)

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
    classification_report, confusion_matrix
)

from NetFoundDataCollator import DataCollatorForFlowClassification
from NetFoundModels import NetfoundFinetuningModel, NetfoundNoPTM, NetfoundFeatureExtractor
from NetFoundTrainer import NetfoundTrainer, NetfoundFeatureExtractorTrainer
from NetfoundConfig import NetfoundConfig, NetFoundTCPOptionsConfig, NetFoundLarge
from NetfoundTokenizer import NetFoundTokenizer
from utils import ModelArguments, CommonDataTrainingArguments, freeze, verify_checkpoint, \
    load_train_test_datasets, load_full_dataset, get_90_percent_cpu_count, get_logger, init_tbwriter, update_deepspeed_config, \
    LearningRateLogCallback

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight

random.seed(42)
logger = get_logger(name=__name__)


@dataclass
class FineTuningDataTrainingArguments(CommonDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    num_labels: int = field(metadata={"help": "number of classes in the datasets"}, default=None)
    problem_type: Optional[str] = field(
        default=None,
        metadata={"help": "Override regression or classification task"},
    )
    p_val: float = field(
        default=0,
        metadata={
            "help": "noise rate"
        },
    )
    netfound_large: bool = field(
        default=False,
        metadata={
            "help": "Use the large configuration for netFound model"
        },
    )
    do_train_feature_extractor: bool = field(
        default=False,
        metadata={"help": "Whether to do feature extractor training."},
    )
    do_feature_extraction: bool = field(
        default=False,
        metadata={"help": "Whether to do feature extraction."},
    )
    do_rf_train: bool = field(
        default=False,
        metadata={"help": "Whether to do random forest training."},
    )
    do_rf_eval: bool = field(
        default=False,
        metadata={"help": "Whether to do random forest eval."},
    )
    do_grid_search: bool = field(
        default=False,
        metadata={"help": "Do grid search for RF hyperparameters."},
    )
    hr_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory with preprocessed hidden representation (features.joblib and labels.joblib files)."},
    )
    n_estimators: int = field(
        default=100,
        metadata={"help": "Number of trees in the random forest."},
    )
    finetuned_base_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Use a finetuned base netfound for the feature extractor rather than the generic base."},
    )
    layers_to_unfreeze: int = field(
        default=0,
        metadata={"help": "Number of layers to unfreeze when training netfound base."},
    )
    do_ensemble: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate using an ensemble of the base and the random forest model."},
    )
    ensemble_rf_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory with pretrained random forest for ensemble."},
    )
    unpoisoned_data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "If this directory is specified, will do train/test split for this dir and for train_dir. Use train set from train_dir and test set from this dir."},
    )


def regression_metrics(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    label_ids = p.label_ids.astype(int)
    return {"loss": np.mean(np.absolute((logits - label_ids)))}


def classif_metrics(p: EvalPrediction, num_classes):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    label_ids = p.label_ids.astype(int)
    weighted_f1 = f1_score(
        y_true=label_ids, y_pred=logits.argmax(axis=1), average="weighted", zero_division=0
    )
    weighted_prec = precision_score(
        y_true=label_ids, y_pred=logits.argmax(axis=1), average="weighted", zero_division=0
    )
    weighted_recall = recall_score(
        y_true=label_ids, y_pred=logits.argmax(axis=1), average="weighted", zero_division=0
    )
    accuracy = accuracy_score(y_true=label_ids, y_pred=logits.argmax(axis=1))
    logger.warning(classification_report(label_ids, logits.argmax(axis=1), digits=5))
    logger.warning(confusion_matrix(label_ids, logits.argmax(axis=1)))
    if num_classes > 3:
        logger.warning(f"top3:{top_k_accuracy_score(label_ids, logits, k=3, labels=np.arange(num_classes))}")
    if num_classes > 5:
        logger.warning(f"top5:{top_k_accuracy_score(label_ids, logits, k=5, labels=np.arange(num_classes))}")
    if num_classes > 10:
        logger.warning(f"top10:{top_k_accuracy_score(label_ids, logits, k=10, labels=np.arange(num_classes))}")

    logger.warning(f"accuracy: {accuracy}")
    logger.warning(f"weighted_prec: {weighted_prec}")
    logger.warning(f"weighted_recall: {weighted_recall}")
    logger.warning(f"weighted_f1: {weighted_f1}")
    
    return {
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "weighted_prec: ": weighted_prec,
        "weighted_recall": weighted_recall,
    }


@record
def main():
    np.set_printoptions(threshold=sys.maxsize) # Don't truncate confusion matrix if we have many classes

    parser = HfArgumentParser(
        (ModelArguments, FineTuningDataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    utils.LOGGING_LEVEL = training_args.get_process_log_level()

    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

    if data_args.do_feature_extraction or data_args.do_train_feature_extractor or data_args.do_ensemble:
        train_dataset, test_dataset = load_train_test_datasets(logger, data_args)
        if "WORLD_SIZE" in os.environ:
            train_dataset = split_dataset_by_node(train_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
            test_dataset = split_dataset_by_node(test_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

    config = NetFoundTCPOptionsConfig if data_args.tcpoptions else NetfoundConfig
    config = config(
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=model_args.num_attention_heads,
        hidden_size=model_args.hidden_size,
        no_meta=data_args.no_meta,
        flat=data_args.flat,
    )
    if data_args.netfound_large:
        config.hidden_size = NetFoundLarge().hidden_size
        config.num_hidden_layers = NetFoundLarge().num_hidden_layers
        config.num_attention_heads = NetFoundLarge().num_attention_heads

    config.pretraining = False
    config.num_labels = data_args.num_labels
    config.problem_type = data_args.problem_type
    testingTokenizer = NetFoundTokenizer(config=config)

    training_config = deepcopy(config)
    training_config.p = data_args.p_val
    training_config.limit_bursts = data_args.limit_bursts
    trainingTokenizer = NetFoundTokenizer(config=training_config)
    additionalFields = None

    if "WORLD_SIZE" in os.environ and training_args.local_rank > 0 and not data_args.streaming:
        logger.warning("Waiting for main process to perform the mapping")
        torch.distributed.barrier()

    params = {
        "batched": True
    }
    if not data_args.streaming:
        params['num_proc'] = data_args.preprocessing_num_workers or get_90_percent_cpu_count()
    
    if data_args.do_feature_extraction or data_args.do_train_feature_extractor or data_args.do_ensemble:
        train_dataset = train_dataset.map(function=trainingTokenizer, **params)
        test_dataset = test_dataset.map(function=testingTokenizer, **params)

    if "WORLD_SIZE" in os.environ and training_args.local_rank == 0 and not data_args.streaming:
        logger.warning("Loading results from main process")
        torch.distributed.barrier()

    data_collator = DataCollatorForFlowClassification(config.max_burst_length)
    # if model_args.model_name_or_path is not None and os.path.exists(
    #         model_args.model_name_or_path
    # ):
    #     logger.warning(f"Using weights from {model_args.model_name_or_path}")
    #     model = freeze(NetfoundFinetuningModel.from_pretrained(
    #         model_args.model_name_or_path, config=config
    #     ), model_args)
    # elif model_args.no_ptm:
    #     model = NetfoundNoPTM(config=config)
    # else:
    #     model = freeze(NetfoundFinetuningModel(config=config), model_args)
    # if training_args.local_rank == 0:
    #     summary(model)

    ### CHANGE TO USE NETFOUND FEATURE EXTRACTOR

    # metrics
    problem_type = data_args.problem_type
    if problem_type == "regression":
        compute_metrics = regression_metrics
    else:
        compute_metrics = lambda p: classif_metrics(p, data_args.num_labels)

    # Compute class weights
    # train_labels = train_dataset["labels"]
    # class_weights = compute_class_weight(
    #     class_weight="balanced",
    #     classes=np.unique(train_labels),
    #     y=train_labels
    # )
    # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(training_args.device)

    # Log smoothed weights
    if data_args.do_feature_extraction or data_args.do_train_feature_extractor or data_args.do_ensemble:
        train_labels = train_dataset["labels"]
        class_counts = np.bincount(train_labels)
        
        smoothing_factor = 1.0  # This is a hyperparameter you can tune
        log_weights = [1.0 / math.log(smoothing_factor + count) for count in class_counts]

        # Normalize the weights so they aren't astronomically large
        # This helps with training stability
        sum_weights = sum(log_weights)
        normalized_weights = [w * (len(class_counts) / sum_weights) for w in log_weights]
        class_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float).to(training_args.device)
    
    # verify_checkpoint(logger, training_args)
      
    # utils.start_gpu_logging(training_args.output_dir)
    # utils.start_cpu_logging(training_args.output_dir)

    if not data_args.do_grid_search:
        # These parameters were tuned by grid search method
        rf_classifier = RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

    if data_args.do_train_feature_extractor:
        logger.warning("*** 1 train netfound feature extractor using default netfound finetuning model ***")

        if os.path.exists(training_args.output_dir):
            logger.warning(f"{training_args.output_dir} already exists - abort as to not overwrite")
            sys.exit()
        
        logger.warning(f"Using weights from {model_args.model_name_or_path}")
        model = freeze(NetfoundFinetuningModel.from_pretrained(
            model_args.model_name_or_path, config=config
        ), model_args)

        # Unfreeze last n hidden layers
        for layer in model.base_transformer.encoder.layer[-data_args.layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Unfreeze final embeddings
        for param in model.base_transformer.encoder.burst_positions.parameters():
            param.requires_grad = True
        for param in model.base_transformer.encoder.flow_positions.parameters():
            param.requires_grad = True
        
        logger.warning(f"Unfroze last {data_args.layers_to_unfreeze} hidden layers and final positional embeddings")
        summary(model)

        model.set_class_weights(class_weights_tensor)

        trainer = NetfoundTrainer(
            model=model,
            extraFields=additionalFields,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=testingTokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
            data_collator=data_collator,
        )  
        
        trainer.train()
        trainer.save_model()


    if data_args.do_feature_extraction:
        logger.warning("*** 2 Use netfound to extract features from training and test sets ***")

        if os.path.exists(data_args.hr_dir):
            logger.warning(f"{data_args.hr_dir} already exists - abort as to not overwrite")
            sys.exit()
        else:
            os.mkdir(data_args.hr_dir)

        logger.warning(f"Using weights from {data_args.finetuned_base_dir}")
        model = freeze(NetfoundFeatureExtractor.from_pretrained(
            data_args.finetuned_base_dir, config=config
        ), model_args)
        # Need to freeze attentive pooling for feature extraction       
        for param in model.attentivePooling.parameters():
            if model_args.freeze_base:
                param.requires_grad = False
        summary(model)

        training_args.eval_strategy = "no"
        training_args.save_strategy = "no"
        trainer = NetfoundFeatureExtractorTrainer(
            model=model,
            extraFields=additionalFields,
            args=training_args,
            tokenizer=testingTokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
            data_collator=data_collator,
        )
        init_tbwriter(training_args.output_dir)
        trainer.add_callback(LearningRateLogCallback(utils.TB_WRITER))

        # This is just using netfound to extract the hidden representation of the input - not making predictions
        logger.warning("extracting features from training set")
        trainer.evaluate(eval_dataset=train_dataset)
        trainer.dump_features(data_args.hr_dir, "train")

        # Reset otherwise the test set will be train + test combined
        trainer.reset_stored_features()

        logger.warning("extracting features from test set")
        trainer.evaluate(eval_dataset=test_dataset)
        trainer.dump_features(data_args.hr_dir, "test")


    if data_args.do_rf_train:
        logger.warning("*** 3 train RF classifier ***")
        features_path = os.path.join(data_args.hr_dir, "train_features.joblib")
        labels_path = os.path.join(data_args.hr_dir, "train_labels.joblib")

        if not os.path.exists(features_path):
            logger.warning(f"{features_path} does not exist")
        
        if not os.path.exists(labels_path):
            logger.warning(f"{labels_path} does not exist")

        if not os.path.exists(training_args.output_dir):
            os.mkdir(training_args.output_dir)

        rf_classifier_path = os.path.join(training_args.output_dir, "rf_classifier.joblib")
        if os.path.exists(rf_classifier_path):
            logger.warning(f"{rf_classifier_path} already exists - abort as to not overwrite")
            sys.exit()

        logger.warning(f"Loading features from {features_path}")
        features = joblib.load(features_path)

        logger.warning(f"Loading labels from {labels_path}")
        labels = joblib.load(labels_path)

        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        logger.warning("Start training")

        rf_classifier.fit(features, labels)

        joblib.dump(rf_classifier, rf_classifier_path)
        logger.warning(f"Classifier saved to {rf_classifier_path}")

        del features
        del labels

    if data_args.do_grid_search:
        logger.warning("*** 4 RF classifier hyperparameter grid search ***")
        features_path = os.path.join(data_args.hr_dir, "train_features.joblib")
        labels_path = os.path.join(data_args.hr_dir, "train_labels.joblib")

        if not os.path.exists(features_path):
            logger.warning(f"{features_path} does not exist")
        
        if not os.path.exists(labels_path):
            logger.warning(f"{labels_path} does not exist")

        if not os.path.exists(training_args.output_dir):
            os.mkdir(training_args.output_dir)

        rf_classifier_path = os.path.join(training_args.output_dir, "rf_classifier.joblib")
        if os.path.exists(rf_classifier_path):
            logger.warning(f"{rf_classifier_path} already exists - abort as to not overwrite")
            sys.exit()

        logger.warning(f"Loading features from {features_path}")
        features = joblib.load(features_path)

        logger.warning(f"Loading labels from {labels_path}")
        labels = joblib.load(labels_path)

        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        logger.warning("Start search")

        rf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [20, 30, None],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring='f1_macro',
            verbose=4
        )
        grid_search.fit(features, labels)

        print(f"Best parameters found: {grid_search.best_params_}")

        joblib.dump(grid_search.best_estimator_, rf_classifier_path)
        logger.warning(f"Best classifer saved to {rf_classifier_path}")

        del features
        del labels

    if data_args.do_rf_eval:
        logger.warning("*** 5 Evaluate ***")

        # Load the trained Random Forest classifier
        rf_classifier_path = os.path.join(training_args.output_dir, "rf_classifier.joblib")
        if not os.path.exists(rf_classifier_path):
            logger.warning(f"{rf_classifier_path} does not exist")

        # Load features and labels
        features_path = os.path.join(data_args.hr_dir, "test_features.joblib")
        labels_path = os.path.join(data_args.hr_dir, "test_labels.joblib")

        if not os.path.exists(features_path):
            logger.warning(f"{features_path} does not exist")
        
        if not os.path.exists(labels_path):
            logger.warning(f"{labels_path} does not exist")

        logger.warning(f"Loading features from {features_path}")
        features = joblib.load(features_path)

        logger.warning(f"Loading labels from {labels_path}")
        labels = joblib.load(labels_path)

        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
 
        logger.warning(f"Loading trained RF classifier from {rf_classifier_path}")
        rf_classifier = joblib.load(rf_classifier_path)

        logger.warning("Start evaluating")

        predictions = rf_classifier.predict_proba(features)
        p = EvalPrediction(
            predictions=predictions,
            label_ids=labels
        )
        classif_metrics(p, data_args.num_labels)

        del features
        del labels

    if data_args.do_ensemble:
        logger.warning("*** 6 Ensemble ***")

        logger.warning(f"Using weights from {data_args.finetuned_base_dir}")
        model = freeze(NetfoundFinetuningModel.from_pretrained(
            data_args.finetuned_base_dir, config=config
        ), model_args)

        summary(model)

        model.set_class_weights(class_weights_tensor)

        training_args.eval_strategy = "no"
        training_args.save_strategy = "no"
        trainer = NetfoundTrainer(
            model=model,
            extraFields=additionalFields,
            args=training_args,
            tokenizer=testingTokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
            data_collator=data_collator,
        )
        init_tbwriter(training_args.output_dir)
        trainer.add_callback(LearningRateLogCallback(utils.TB_WRITER))

         # Load the trained Random Forest classifier
        rf_classifier_path = os.path.join(data_args.ensemble_rf_dir, "rf_classifier.joblib")
        if not os.path.exists(rf_classifier_path):
            logger.warning(f"{rf_classifier_path} does not exist")

        # Load features and labels
        features_path = os.path.join(data_args.hr_dir, "test_features.joblib")
        labels_path = os.path.join(data_args.hr_dir, "test_labels.joblib")

        if not os.path.exists(features_path):
            logger.warning(f"{features_path} does not exist")
        
        if not os.path.exists(labels_path):
            logger.warning(f"{labels_path} does not exist")

        logger.warning(f"Loading features from {features_path}")
        features = joblib.load(features_path)

        logger.warning(f"Loading labels from {labels_path}")
        labels = joblib.load(labels_path)

        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
 
        logger.warning(f"Loading trained RF classifier from {rf_classifier_path}")
        rf_classifier = joblib.load(rf_classifier_path)

        logger.warning("Get predictions from finetuned base model")
        output = trainer.predict(test_dataset=test_dataset)
        base_predictions = output.predictions

        # Save these predictions because they take ages to compute, then we can try different weightings later
        base_predictions_path = os.path.join(training_args.output_dir, "base_predictions.joblib")
        joblib.dump(base_predictions, base_predictions_path)

        logger.warning("Get predictions from RF hybrid model")
        rf_predictions = rf_classifier.predict_proba(features)

        # Weighted average of predictions
        final_predictions = (base_predictions * 0.25) + (rf_predictions * 0.75)
        p = EvalPrediction(
            predictions=final_predictions,
            label_ids=labels
        )
        classif_metrics(p, data_args.num_labels)

if __name__ == "__main__":
    main()