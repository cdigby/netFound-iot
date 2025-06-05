#!/bin/bash

set -e
set +x

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_folder output_folder"
    exit 1
fi

# Get the directory where the current script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

input_folder="$1"
output_folder="$2"

mkdir -p "$output_folder"

# Check if 2_pcap_splitting script exists in the same directory as the current script
pcap_splitting_script="$script_dir/2_pcap_splitting"
if [ ! -f "$pcap_splitting_script" ]; then
    echo "Error: 2_pcap_splitting script not found in $script_dir, please run make all from the project root directory"
    exit 1
fi

find "$input_folder" -type f | parallel "mkdir -p $output_folder/{/.} && $pcap_splitting_script -f {} -o $output_folder/{/.} -m connection"
