#!/bin/bash

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

[ -z "$output" ] && echo "Please enter a output path with the [-o] flag!" && exit 2;
[ -z "$model" ] && echo "Please enter a model with the [-m] flag!" && exit 2;

echo "Evaluating $model on $file... Output will be written to $output"

sudo -u mitty PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/mitty_experiment/ner_tagger_mitty.py \
	--data_dir ./data \
	--eval_file  "$file" \
	--eval_output_file  ./output/mitty_experiment/"$output" \
	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
	--charlm \
	--charlm_shorthand 1billion \
	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
	--mode predict \
	--save_dir ./models/mitty_experiment/"$model" \
	--shorthand "$pre""$model" --cpu
