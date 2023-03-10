#!/bin/bash

output="./output/output_attn_l.tsv"
file="./stanza/TOC_Utility/Processed_Data/synth_combined.test.json"
pre="en_"
while getopts o:i:m:p: flag
do
    case "${flag}" in
        o) output=${OPTARG};;
        i) file=${OPTARG};;
        m) model=${OPTARG};;
        p) pre=${OPTARG};;
    esac
done

[ -z "$model" ] && echo "Please enter a model with the [-m] flag!" && exit 2;

echo "Evaluating $model on $file... Output will be written to $output"

python3 stanza/models/mitty_experiment/ner_tagger_mitty.py \
	--data_dir ./data \
	--eval_file  "$file" \
	--eval_output_file  "$output" \
	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
	--charlm \
	--charlm_shorthand 1billion \
	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
	--mode predict \
	--save_dir ./models/data_extractor/"$model" \
	--shorthand "$pre""$model" --cpu
