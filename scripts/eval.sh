#!/bin/bash

# NO NEED TO CHANGE DEFAULT ARGS, JUST PASS IN YOUR RELEVANT ONES AT COMMAND LINE
output="./output/output.tsv"
file="./stanza/TOC_Utility/Processed_Data/synth_combined.test.json"
pre="en_"
suff="_dataextractor"
folder="data_extractor"
args=""
while getopts e:o:i:m:p:s:f: flag
do
    case "${flag}" in
        o) output=${OPTARG};;
        i) file=${OPTARG};;
        m) model=${OPTARG};;
        p) pre=${OPTARG};;
        s) suff=${OPTARG};;
        f) folder=${OPTARG};;
        e) args=${OPTARG};;
    esac
done

[ -z "$model" ] && echo "Please enter a model with the [-m] flag!" && exit 2;

echo "Evaluating $model on $file... Output will be written to $output"

python3 stanza/models/data_extractor.py \
	--data_dir ./data \
	--eval_file  "$file" \
	--eval_output_file  "$output" \
	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
	--charlm \
	--charlm_shorthand 1billion \
	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
	--mode predict \
	--save_dir ./models/"$folder"/"$model" \
	--save_name "$pre""$model""$suff".pt $args
