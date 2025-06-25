#!/bin/bash

set +x

# if [ "$#" -ne 2 ]; then
#     echo "Usage: $0 input_folder output_folder"
#     exit 1
# fi

# Get the directory where the current script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# pcap_size=20

input_folder=/mnt/extra/data/cic-iot-2023-pcap/raw
output_folder=/mnt/extra/data/iot2023-8class-big-benign/raw

mkdir -p "$output_folder"

# for i in $(seq 0 7);
# do
#   echo $i
#   find "$input_folder/$i" -type f | parallel "mkdir -p $output_folder/$i && tcpdump -r {} -w $output_folder/$i/{/} -C $pcap_size -Z root && find \"$output_folder/$i\" -type f | grep -P \"(.pcap)[0-9]+\" | xargs -d\"\n\" rm"
# done

# Benign
i=0
size=600
find "$input_folder/0" -type f | parallel "mkdir -p $output_folder/$i && tcpdump -r {} -w $output_folder/$i/{/} -C $size -Z root && find \"$output_folder/$i\" -type f | grep -P \"(.pcap)[0-9]+\" | xargs -d\"\n\" rm"

# DDoS SYN Flood
i=1
size=10
find "$input_folder/11" -type f | parallel "mkdir -p $output_folder/$i && tcpdump -r {} -w $output_folder/$i/{/} -C $size -Z root && find \"$output_folder/$i\" -type f | grep -P \"(.pcap)[0-9]+\" | xargs -d\"\n\" rm"

# DoS SYN Flood
i=2
size=10
find "$input_folder/19" -type f | parallel "mkdir -p $output_folder/$i && tcpdump -r {} -w $output_folder/$i/{/} -C $size -Z root && find \"$output_folder/$i\" -type f | grep -P \"(.pcap)[0-9]+\" | xargs -d\"\n\" rm"

# Recon Host Discovery
i=3
size=10
find "$input_folder/26" -type f | parallel "mkdir -p $output_folder/$i && tcpdump -r {} -w $output_folder/$i/{/} -C $size -Z root && find \"$output_folder/$i\" -type f | grep -P \"(.pcap)[0-9]+\" | xargs -d\"\n\" rm"

# Command Injection
i=4
size=200
find "$input_folder/3" -type f | parallel "mkdir -p $output_folder/$i && tcpdump -r {} -w $output_folder/$i/{/} -C $size -Z root && find \"$output_folder/$i\" -type f | grep -P \"(.pcap)[0-9]+\" | xargs -d\"\n\" rm"

# Dictionary Brute Force
i=5
size=200
find "$input_folder/16" -type f | parallel "mkdir -p $output_folder/$i && tcpdump -r {} -w $output_folder/$i/{/} -C $size -Z root && find \"$output_folder/$i\" -type f | grep -P \"(.pcap)[0-9]+\" | xargs -d\"\n\" rm"

# DNS Spoofing
i=6
size=200
find "$input_folder/17" -type f | parallel "mkdir -p $output_folder/$i && tcpdump -r {} -w $output_folder/$i/{/} -C $size -Z root && find \"$output_folder/$i\" -type f | grep -P \"(.pcap)[0-9]+\" | xargs -d\"\n\" rm"

# Mirai udpplain
i=7
size=2000
find "$input_folder/24" -type f | parallel "mkdir -p $output_folder/$i && tcpdump -r {} -w $output_folder/$i/{/} -C $size -Z root && find \"$output_folder/$i\" -type f | grep -P \"(.pcap)[0-9]+\" | xargs -d\"\n\" rm"
