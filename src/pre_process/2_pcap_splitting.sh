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

# Need to increase open file limit or SplitCap will crash on big pcap files
ulimit -n 100000

find "$input_folder" -type f | parallel "mkdir -p $output_folder/{/.} && mono  $script_dir/SplitCap.exe -r {} -o $output_folder/{/.} -s session"
