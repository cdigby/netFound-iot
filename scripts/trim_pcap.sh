#!/bin/bash

set +x

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_folder output_folder"
    exit 1
fi

# Get the directory where the current script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pcap_size=20

input_folder="$1"
output_folder="$2"

mkdir -p "$output_folder"

for i in $(seq 0 7);
do
  echo $i
  find "$input_folder/$i" -type f | parallel "mkdir -p $output_folder/$i && tcpdump -r {} -w $output_folder/$i/{/} -C $pcap_size -Z root && find \"$output_folder/$i\" -type f | grep -P \"(.pcap)[0-9]+\" | xargs -d\"\n\" rm"
done