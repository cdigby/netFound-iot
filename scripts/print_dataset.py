from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import logging, sys

# Ensure a filename is provided as the first argument
if len(sys.argv) < 2:
    print("Usage: python script_name.py <arrow_file>")
    sys.exit(1)
 
# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
 
# Add StreamHandler to output logs to the notebook
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
 
# Define dataset path (update this if necessary)
dataset_path = sys.argv[1]
 
logger.info("Loading dataset...")
 
# Load dataset (assuming it's stored as Arrow files)
# dataset = load_dataset("arrow", data_dir=dataset_path, split="train")
dataset = Dataset.from_file(dataset_path)
 
# Convert dataset to Pandas DataFrame
df = dataset.to_pandas()
 
# Print available columns
logger.info(f"Available columns: {df.columns.tolist()}")

#Print number of rows
logger.info(f"Number of rows: {len(df)}")