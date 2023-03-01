#!/bin/bash

# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth2.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth2.dev.json \
# 	--eval_output_file  ./output/synth2_vanilla_out.tsv \
# 	--mode train \
# 	--save_dir ./models/data_extractor/synth2_vanilla \
# 	--shorthand en_synth2_vanilla \
# 	--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
# 	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 	--tensorboard | tee ./training.log

# train from BERT
sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/ner_tagger.py \
	--data_dir ./data \
	--train_file ./stanza/TOC_Utility/Processed_Data/synth2.train.json \
	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth2.dev.json \
	--mode train \
	--save_dir ./models/data_extractor/synth2_vanilla_from_bert \
	--shorthand en_synth2_vanilla_from_bert \
	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
	--charlm \
	--charlm_shorthand 1billion \
	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
	--tensorboard

# train only class layers
sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
	--data_dir ./data \
	--train_file ./stanza/TOC_Utility/Processed_Data/synth2.train.json \
	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth2.dev.json \
	--mode train \
	--train_classifier_only \
	--save_dir ./models/data_extractor/synth2_vanilla_clf_only \
	--shorthand en_synth2_vanilla_clf_only \
	--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
	--charlm \
	--charlm_shorthand 1billion \
	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
	--tensorboard

echo "WARNING! SYSTEM WILL SHUTDOWN IN 60 SECONDS UNLESS CTRL-C INTERRUPT..."
sleep 60
shutdown now
