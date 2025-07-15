from dataclasses import dataclass, field
from typing import Optional
import datasets
import transformers
import logging
import os
import json
import threading
import subprocess
import torch
import time
import socket
import psutil

from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
from tqdm import tqdm

LOGGING_LEVEL = logging.WARNING
TB_WRITER: Optional[SummaryWriter] = None


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    metaFeatures: int = field(
        default=4,
        metadata={"help": "number of metadata fields."},
    )
    num_hidden_layers: int = field(
        default=12,
        metadata={"help": "Number of hidden layers."},
    )
    num_attention_heads: int = field(
        default=12,
        metadata={"help": "Number of attention heads."},
    )
    hidden_size: int = field(
        default=768,
        metadata={"help": "Hidden size."},
    )
    no_ptm: bool = field(
        default=False,
        metadata={"help": "If True, use NoPTM model (only for fine-tuning)."},
    )
    freeze_flow_encoder: bool = field(
        default=False,
        metadata={"help": "Freeze flow encoders"},
    )
    freeze_burst_encoder: bool = field(
        default=False,
        metadata={"help": "Freeze burst encoders"},
    )
    freeze_embeddings: bool = field(
        default=False,
        metadata={"help": "Freeze embeddings"},
    )
    freeze_base: bool = field(
        default=False,
        metadata={"help": "Freeze base model"},
    )


@dataclass
class CommonDataTrainingArguments:
    train_dir: Optional[str] = field(
        metadata={"help": "Directory with training data (Apache Arrow files)"})
    test_dir: Optional[str] = field(default=None, metadata={
        "help": "Directory with testing data (Apache Arrow files)"})
    no_meta: bool = field(
        default=False,
        metadata={"help": "no meta fields"},
    )
    flat: bool = field(
        default=False,
        metadata={"help": "no cross burst encoder"},
    )
    limit_bursts: bool = field(
        default=False,
        metadata={"help": "limit_bursts"},
    )
    validation_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory with optional input evaluation data to evaluate the perplexity on (Apache Arrow files)"},
    )
    validation_split_percentage: Optional[int] = field(
        default=30,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"}
    )
    data_cache_dir: Optional[str] = field(
        default="/tmp",
        metadata={"help": "Where to store the dataset cache."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    max_bursts: int = field(
        default=12,
        metadata={
            "help": "The maximum number of sentences after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    max_seq_length: Optional[int] = field(
        default=1296 + 12,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[float] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to load dataset in the streaming mode."},
    )
    tcpoptions: bool = field(
        default=False,
        metadata={"help": "Whether the data contains TCP options."},
    )


def freeze(model, model_args):
    for name, param in model.base_transformer.named_parameters():
        if model_args.freeze_flow_encoder and (
                "flow_encoder" in name or ("encoder" in name and "position_embeddings" in name)):
            param.requires_grad = False
        if model_args.freeze_burst_encoder and "burst_encoder" in name:
            param.requires_grad = False
        if model_args.freeze_embeddings and (name.startswith("embed") or name.startswith("seg_embed")):
            param.requires_grad = False
        if model_args.freeze_base:
            param.requires_grad = False
    return model


def get_logger(name):
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(LOGGING_LEVEL)
    datasets.utils.logging.set_verbosity(LOGGING_LEVEL)
    transformers.utils.logging.set_verbosity(LOGGING_LEVEL)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    return logger


def verify_checkpoint(logger, training_args):
    if not training_args.resume_from_checkpoint:
        folders = set(os.listdir(training_args.output_dir)) - {"runs"}
        if len(folders) > 0:
            if training_args.local_rank == 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overwrite it."
                )
    else:
        if training_args.local_rank == 0:
            resume_from_checkpoint = training_args.resume_from_checkpoint if isinstance(training_args.resume_from_checkpoint, str) else get_last_checkpoint(training_args.output_dir)
            logger.warning(
                f"Checkpoint detected, resuming training at {resume_from_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


def get_90_percent_cpu_count():
    return max(1, int(os.cpu_count() * 0.9))


def load_train_test_datasets(logger, data_args):
    logger.warning("Loading datasets")
    if data_args.test_dir is None:
        data_args.test_dir = data_args.train_dir
        train_split = f"train[{data_args.validation_split_percentage}%:]"
        test_split = f"train[:{data_args.validation_split_percentage}%]"
    else:
        train_split = "train"
        test_split = "train"

    train_dataset = load_dataset(
        "arrow",
        data_dir=data_args.train_dir,
        split=train_split,
        cache_dir=data_args.data_cache_dir,
        streaming=data_args.streaming,
    )

    test_dataset = load_dataset(
        "arrow",
        data_dir=data_args.test_dir,
        split=test_split,
        cache_dir=data_args.data_cache_dir,
        streaming=data_args.streaming,
    )

    if data_args.max_eval_samples is not None:
        test_dataset = test_dataset.select(
            range(min(test_dataset.shape[0], data_args.max_eval_samples))
        )
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(train_dataset.shape[0], data_args.max_train_samples))
        )

    if not data_args.streaming:
        total_bursts_train = [0] * len(train_dataset)
        total_bursts_test = [0] * len(test_dataset)
    else:
        total_bursts_train = defaultdict(lambda: 0)
        total_bursts_test = defaultdict(lambda: 0)

    train_dataset = train_dataset.add_column("total_bursts", total_bursts_train)
    test_dataset = test_dataset.add_column("total_bursts", total_bursts_test)

    return train_dataset, test_dataset

def create_scalar_features(df):
    """
    Creates a temporary, flat numerical feature set from the original DataFrame.
    This is used ONLY for calculating distances to find neighbors.
    """
    print("Creating scalar features")
    scalar_df = pd.DataFrame()
    
    # Add scalar features directly
    scalar_df['flow_duration'] = df['flow_duration']
    scalar_df['protocol'] = df['protocol']

    # Calculate the ratio of outgoing (True) bursts
    scalar_df['directions_outgoing_ratio'] = df['directions'].apply(
        lambda x: np.mean(x) if len(x) > 0 else 0.5
    )

    # Calculate statistical features of lists
    list_features = ['bytes', 'iats', 'counts']
    for col in tqdm(list_features):
        scalar_df[f'{col}_mean'] = df[col].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
        scalar_df[f'{col}_std'] = df[col].apply(lambda x: np.std(x) if len(x) > 1 else 0)
        scalar_df[f'{col}_sum'] = df[col].apply(np.sum)
        scalar_df[f'{col}_min'] = df[col].apply(lambda x: np.min(x) if len(x) > 0 else 0)
        scalar_df[f'{col}_max'] = df[col].apply(lambda x: np.max(x) if len(x) > 0 else 0)
        scalar_df[f'{col}_q25'] = df[col].apply(lambda x: np.quantile(x, 0.25) if len(x) > 0 else 0)
        scalar_df[f'{col}_q50'] = df[col].apply(lambda x: np.quantile(x, 0.5) if len(x) > 0 else 0)
        scalar_df[f'{col}_q75'] = df[col].apply(lambda x: np.quantile(x, 0.75) if len(x) > 0 else 0)

    scalar_df['num_bursts'] = df['directions'].apply(len)

    print("Created scalar features")
    return scalar_df

def adaptive_oversample(df, target_classes, k_neighbors=20):
    """
    Augments the dataset by adaptively duplicating the "hardest" minority samples.
    Inspired by ADASYN
    
    Args:
        df (pd.DataFrame): The original dataframe.
        target_classes (list): A list of class labels (as strings) to augment.
        k_neighbors (int): Number of neighbors to consider when assessing difficulty.
    """
    print("--- Starting Adaptive Weighted Over-sampling ---")
    
    # Create the temporary scalar feature set for finding neighbors
    scalar_features = create_scalar_features(df)
    
    # Fit a NearestNeighbors model on the entire dataset's scalar features
    nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto').fit(scalar_features)
    
    new_samples = []
    
    # Determine the target number of samples for minority classes (usually the majority class count)
    try:
        majority_class_count = df['labels'].value_counts().max()
    except ValueError: # Happens if dataframe is empty
        return df

    samples_to_add = []
    
    for target_class in target_classes:
        class_df = df[df['labels'] == target_class]
        
        if len(class_df) == 0:
            print(f"No samples found for class '{target_class}'. Skipping.")
            continue
            
        class_indices = class_df.index
        
        # --- Calculate difficulty scores LOCALLY for this class ---
        difficulty_scores = []
        print(f"Calculating difficulty scores for {len(class_indices)} samples in class '{target_class}'...")
        for index in class_indices:
            minority_label = df.loc[index, 'labels']
            
            # Find the k nearest neighbors in the full dataset
            neighbor_indices = nn_model.kneighbors(scalar_features.loc[[index]], return_distance=False)
            
            # Get the labels of these neighbors
            neighbor_labels = df.loc[neighbor_indices[0], 'labels']
            
            # Calculate the ratio of neighbors from OTHER classes
            num_different_class = sum(1 for label in neighbor_labels if label != minority_label)
            difficulty = num_different_class / k_neighbors
            difficulty_scores.append(difficulty)

        # --- Perform weighted duplication based on local weights ---
        # num_to_generate = majority_class_count - len(class_df)
        num_to_generate = 10000
        if num_to_generate <= 0:
            print(f"Class '{target_class}' is already balanced. Skipping.")
            continue

        # Normalize the difficulty scores to create a probability distribution for this class
        total_difficulty = sum(difficulty_scores)
        if total_difficulty == 0:
            print(f"All samples in class '{target_class}' are 'easy'. Using uniform duplication.")
            class_weights = None # Fallback to uniform probability
        else:
            class_weights = [score / total_difficulty for score in difficulty_scores]

        print(f"Generating {num_to_generate} new samples for class '{target_class}'...")
        # Select which samples to duplicate based on the calculated weights.
        # The lengths of `class_indices` and `class_weights` are now guaranteed to match.
        duplicated_indices = random.choices(
            class_indices.tolist(), 
            weights=class_weights, 
            k=num_to_generate
        )
        
        samples_to_add.extend(df.loc[duplicated_indices].to_dict('records'))

    # Combine the original dataframe with the new samples
    if not samples_to_add:
        print("No new samples were generated.")
        return df

    augmented_df = pd.concat([df, pd.DataFrame(samples_to_add)], ignore_index=True)
    
    print("Augmentation complete.")
    print("-" * 30 + "\n")
    return augmented_df


def load_full_dataset(logger, data_args):
    logger.warning(f"Loading full dataset from {data_args.train_dir}")
    full_dataset = load_dataset(
        "arrow",
        data_dir=data_args.train_dir,
        split="train",
        cache_dir=data_args.data_cache_dir,
        streaming=data_args.streaming,
    )

    if not data_args.streaming:
        total_bursts_full = [0] * len(full_dataset)
    else:
        total_bursts_full = defaultdict(lambda: 0)
    
    full_dataset = full_dataset.add_column("total_bursts", total_bursts_full)

    return full_dataset


def initialize_model_with_deepspeed(logger, training_args, get_model):
    '''
    here we do only specific init if stage 3 is used, otherwise huggingface trainer will do the rest
    '''
    import deepspeed
    import base64
    logger.warning("Initializing deepspeed-optimized model")
    # only if stage 3
    if training_args.deepspeed.endswith(".json"):
        with open(training_args.deepspeed, "r") as f:
            deepspeed_config = json.load(f)
    else:
        deepspeed_config = training_args.deepspeed
        # unbase64
        deepspeed_config = json.loads(base64.b64decode(deepspeed_config).decode("utf-8"))

    is_stage_3 = deepspeed_config.get("zero_optimization", {}).get("stage", 0) == 3
    with deepspeed.zero.Init(enabled=is_stage_3):
        model = get_model()
    optimizers = (None, None)
    return model, optimizers


def init_tbwriter(output_dir=".") -> None:
    global TB_WRITER
    current_time = time.strftime("%b%d_%H-%M-%S", time.localtime())
    if not torch.cuda.is_available():
        TB_WRITER = SummaryWriter(os.path.join(output_dir, "runs", current_time + "_" + socket.gethostname() + f"_pid{os.getpid()}_custom_metrics"))
        return
    TB_WRITER = SummaryWriter(os.path.join(output_dir, "runs", current_time + "_" + socket.gethostname() + f"_gpu{torch.cuda.current_device()}_custom_metrics"))

def get_gpu_utilization(gpu_id):
    """Fetch GPU utilization using nvidia-smi for the given GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        utilization = int(result.stdout.strip())
        return utilization
    except Exception as e:
        get_logger(__name__).error(f"Error fetching GPU utilization: {e}")
        return 0


def log_gpu_stats(gpu_id, output_dir, interval=10):
    """
    Log GPU utilization and memory usage for the assigned GPU to TensorBoard every `interval` seconds.
    """
    if not torch.cuda.is_available():
        get_logger(__name__).error("No GPU found.")
        return
    
    current_time = time.strftime("%b%d_%H-%M-%S", time.localtime())
    writer = SummaryWriter(os.path.join(output_dir, "runs", current_time + "_" + socket.gethostname() + f"_gpu{gpu_id}"))

    while True:
        # Get GPU stats for the current process's assigned GPU
        device = torch.device(f"cuda:{gpu_id}")
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # In GB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # In MB
        memory_free = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3) - memory_reserved

        # Get GPU utilization using nvidia-smi
        utilization = get_gpu_utilization(gpu_id)

        # Log to TensorBoard
        writer.add_scalar(f"GPU/Memory Allocated (GB)", memory_allocated, time.time())
        writer.add_scalar(f"GPU/Memory Reserved (GB)", memory_reserved, time.time())
        writer.add_scalar(f"GPU/Memory Free (GB)", memory_free, time.time())
        writer.add_scalar(f"GPU/Utilization (%)", utilization, time.time())

        # Sleep before logging the next set of stats
        time.sleep(interval)

def start_gpu_logging(output_dir="."):
    """
    Start logging GPU stats to TensorBoard for the current process's assigned GPU.
    """
    if not torch.cuda.is_available():
        get_logger(__name__).error("No GPU found.")
        return

    gpu_id = torch.cuda.current_device()

    # Start logging GPU stats in a separate thread
    gpu_stats_thread = threading.Thread(target=log_gpu_stats, args=(gpu_id, output_dir))
    gpu_stats_thread.daemon = True
    gpu_stats_thread.start()

def log_cpu_stats(output_dir, interval=10):
    current_time = time.strftime("%b%d_%H-%M-%S", time.localtime())
    writer = SummaryWriter(os.path.join(output_dir, "runs", current_time + "_" + socket.gethostname() + f"_cpu_metrics"))

    while True:
        try:
            cpu_load = psutil.cpu_percent(interval=None)
            writer.add_scalar(f"CPU/Utilization %", psutil.cpu_percent(interval=None), time.time())
        except Exception as e:
            get_logger(__name__).error(f"Error fetching CPU utilization: {e}")
            return 0
        
        time.sleep(interval)

def start_cpu_logging(output_dir="."):
    """
    Start logging overall CPU stats to TensorBoard.
    """
    # do it only for a single process per node
    if os.environ.get("SLURM_LOCALID", "-1") != "0":
        return

    cpu_stats_thread = threading.Thread(target=log_cpu_stats, args=(output_dir,))
    cpu_stats_thread.daemon = True
    cpu_stats_thread.start()

def update_deepspeed_config(training_args):
    if training_args.deepspeed is not None and training_args.deepspeed.endswith(".json"):
        with open(training_args.deepspeed, "r") as f:
            training_args.deepspeed = json.load(f)
        if "tensorboard" in training_args.deepspeed:
            training_args.deepspeed["tensorboard"]["output_path"] = training_args.output_dir
            training_args.deepspeed["tensorboard"]["job_name"] = os.environ.get("SLURM_JOB_NAME", "local")
    return training_args

class LearningRateLogCallback(TrainerCallback):
    def __init__(self, tb_writer):
        self.tb_writer = tb_writer

    def on_step_end(self, args, state, control, **kwargs):
        # The optimizer is passed as a keyword argument
        optimizer = kwargs.get('optimizer')
        if optimizer is not None:
            # If you have multiple parameter groups, you can log each group’s LR
            for i, param_group in enumerate(optimizer.param_groups):
                self.tb_writer.add_scalar(f"train/learning_rate/group_{i}", param_group['lr'], state.global_step)
        return control