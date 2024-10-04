#!/bin/bash

# Input model type and index (i)
model_type=$1
index=$2
gpu_index=$3
overwrite=$4

# Validate input
if [[ -z "$model_type" || -z "$index" || -z "$gpu_index" || -z "$overwrite" ]]; then
    echo "Usage: $0 <model_type> <row_index> <gpu_index> <overwrite>"
    exit 1
fi

# File name based on model type
csv_file="configs/${model_type}_configs.csv"

# Check if file exists
if [ ! -f "$csv_file" ]; then
    echo "Error: File '$csv_file' does not exist."
    exit 1
fi

# Read the header to get column names
IFS=',' read -r -a headers < $csv_file

# Replace underscores with hyphens in headers
for i in "${!headers[@]}"; do
    headers[$i]=${headers[$i]//_/-}
done

# Get the i-th row; adjust index by 1 because head/tail commands are 1-based
row=$(tail -n +2 "$csv_file" | head -n $index | tail -n 1)

# Split the row into an array
IFS=',' read -r -a values <<< "$row"

# Check if we actually got a row
if [ -z "$row" ]; then
    echo "Error: Row $index does not exist in $csv_file."
    exit 1
fi

# Construct the argument string for the python command
args=""
for i in "${!headers[@]}"; do
    args+="--${headers[$i]} ${values[$i]} "
done

# Execute the Python script with the arguments
echo "Running command: python fit_model.py $model_type $args -g $gpu_index --seed-everything --wandb-overwrite $overwrite --use-wandb"
python fit_model.py $model_type $args -g $gpu_index --seed-everything --wandb-overwrite $overwrite --use-wandb 
