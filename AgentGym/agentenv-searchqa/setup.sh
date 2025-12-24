#!/bin/bash

# Download data from Hugging Face
# This script sets up the environment for the SearchQA dataset by downloading and processing the data.
# Ensure the script is run from the correct directory
if [[ ! -d "./scripts" ]]; then
    echo "Please run this script from the agentenv-searchqa directory."
    exit 1
fi
# Set the Hugging Face endpoint 
export HF_ENDPOINT="https://hf-mirror.com"
# Create directory to save the downloaded data
local_path=$(dirname "$(readlink -f "$0")")
save_path=${local_path}/retrieve_data
mkdir -p $save_path

# Download the dataset using download.py script
echo "Starting download..."
python ./scripts/download.py --save_path $save_path
echo "Download complete. Merging files..."

# Merge the downloaded files into a single index file
cat $save_path/part_* > $save_path/e5_Flat.index
rm $save_path/part_*

# Decompress the downloaded file
gzip -d $save_path/wiki-18.jsonl.gz

# Load datasets
WORK_DIR=$(dirname "$(readlink -f "$0")")
LOCAL_DIR=$WORK_DIR/agentenv_searchqa/queries
## process multiple dataset search format train file
DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA

## process multiple dataset search format test file
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA

