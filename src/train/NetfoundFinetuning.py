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
from NetFoundTrainer import NetfoundTrainer
from NetfoundConfig import NetfoundConfig, NetFoundTCPOptionsConfig, NetFoundLarge
from NetfoundTokenizer import NetFoundTokenizer
from utils import ModelArguments, CommonDataTrainingArguments, freeze, verify_checkpoint, \
    load_train_test_datasets, load_full_dataset, get_90_percent_cpu_count, get_logger, init_tbwriter, update_deepspeed_config, \
    LearningRateLogCallback

import joblib
from sklearn.ensemble import RandomForestClassifier

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
    hr_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory with preprocessed hidden representation (features.joblib and labels.joblib files)."},
    )
    n_estimators: int = field(
        default=100,
        metadata={"help": "Number of trees in the random forest."},
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

    if data_args.do_feature_extraction:
        full_dataset = load_full_dataset(logger, data_args)
    # train_dataset, test_dataset = load_train_test_datasets(logger, data_args)
    if "WORLD_SIZE" in os.environ:
        full_dataset = split_dataset_by_node(full_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        # train_dataset = split_dataset_by_node(train_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        # test_dataset = split_dataset_by_node(test_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

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
    
    if data_args.do_feature_extraction:
        full_dataset = full_dataset.map(function=trainingTokenizer, **params)
    # train_dataset = train_dataset.map(function=trainingTokenizer, **params)
    # test_dataset = test_dataset.map(function=testingTokenizer, **params)

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
    if data_args.do_feature_extraction:
        if model_args.model_name_or_path is not None and os.path.exists(
                model_args.model_name_or_path
        ):
            logger.warning(f"Using weights from {model_args.model_name_or_path}")
            model = freeze(NetfoundFeatureExtractor.from_pretrained(
                model_args.model_name_or_path, config=config
            ), model_args)
        elif model_args.no_ptm:
            model = NetfoundNoPTM(config=config)
        else:
            model = freeze(NetfoundFeatureExtractor(config=config), model_args)
        if training_args.local_rank == 0:
            summary(model)

    # metrics
    problem_type = data_args.problem_type
    if problem_type == "regression":
        compute_metrics = regression_metrics
    else:
        compute_metrics = lambda p: classif_metrics(p, data_args.num_labels)

    # trainer = NetfoundTrainer(
    #     model=model,
    #     extraFields=additionalFields,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=test_dataset,
    #     tokenizer=testingTokenizer,
    #     compute_metrics=compute_metrics,
    #     callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
    #     data_collator=data_collator,
    # )
    if data_args.do_feature_extraction:
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
        utils.start_gpu_logging(training_args.output_dir)
        utils.start_cpu_logging(training_args.output_dir)

    # verify_checkpoint(logger, training_args)

    # train_dataloader = trainer.get_train_dataloader()

    rf_classifier = RandomForestClassifier(
        n_estimators=data_args.n_estimators,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    if data_args.do_feature_extraction:
        logger.warning("*** 1 Use netfound as feature extractor ***")

        if os.path.exists(data_args.hr_dir):
            logger.warning(f"{data_args.hr_dir} already exists - abort as to not overwrite")
            sys.exit()
        else:
            os.mkdir(data_args.hr_dir)

        # This is just using netfound to extract the hidden representation of the input - not making predictions
        trainer.evaluate(eval_dataset=full_dataset)
        
        logger.warning("*** Save features ***")

        # And then we save to file to use later
        trainer.dump_features(data_args.hr_dir)

    if data_args.do_rf_train:
        logger.warning("*** 2 train RF classifier ***")
        features_path = os.path.join(data_args.hr_dir, "features.joblib")
        labels_path = os.path.join(data_args.hr_dir, "labels.joblib")

        if not os.path.exists(features_path):
            logger.warning(f"{features_path} does not exist")
        
        if not os.path.exists(labels_path):
            logger.warning(f"{labels_path} does not exist")

        rf_classifier_path = os.path.join(training_args.output_dir, "rf_classifier.joblib")
        if os.path.exists(rf_classifier_path):
            logger.warning(f"{rf_classifier_path} already exists - abort as to not overwrite")
            sys.exit()

        if not os.path.exists(training_args.output_dir):
            os.mkdir(training_args.output_dir)

        logger.warning(f"Loading features from {features_path}")
        features = joblib.load(features_path).detach().cpu().numpy()

        logger.warning(f"Loading labels from {labels_path}")
        labels = joblib.load(labels_path).detach().cpu().numpy()

        logger.warning("Start training")

        # Take first 80% of the dataset for training
        split = int(len(features) * 0.8)
        rf_classifier.fit(features[0:split], labels[0:split])

        logger.warning("Save RF classifier...")
        joblib.dump(rf_classifier, rf_classifier_path)
        logger.warning(f"Classifier saved to {rf_classifier_path}")

        del features
        del labels

    if data_args.do_rf_eval:
        logger.warning("*** 3 Evaluate ***")

        # Load the trained Random Forest classifier
        rf_classifier_path = os.path.join(training_args.output_dir, "rf_classifier.joblib")
        if not os.path.exists(rf_classifier_path):
            logger.warning(f"{rf_classifier_path} does not exist")

        # Load features and labels
        features_path = os.path.join(data_args.hr_dir, "features.joblib")
        labels_path = os.path.join(data_args.hr_dir, "labels.joblib")

        if not os.path.exists(features_path):
            logger.warning(f"{features_path} does not exist")
        
        if not os.path.exists(labels_path):
            logger.warning(f"{labels_path} does not exist")

        logger.warning(f"Loading features from {features_path}")
        features = joblib.load(features_path).detach().cpu().numpy()

        logger.warning(f"Loading labels from {labels_path}")
        labels = joblib.load(labels_path).detach().cpu().numpy()
 
        logger.warning(f"Loading trained RF classifier from {rf_classifier_path}")
        rf_classifier = joblib.load(rf_classifier_path)

        logger.warning("Start evaluating")

        # Take last 20% of the dataset for evaluation    
        split = int(len(features) * 0.8)
        predictions = rf_classifier.predict_proba(features[split:len(features)])
        p = EvalPrediction(
            predictions=predictions,
            label_ids=labels[split:len(features)]
        )
        classif_metrics(p, data_args.num_labels)

        del features
        del labels
        

if __name__ == "__main__":
    main()