#!/bin/bash

# train transformer (with grid search)
for nh in 4 8 16
do
	for nt in 4 6 8
	do
		sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
			--data_dir ./data \
			--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
			--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
			--mode train \
			--save_dir ./models/data_extractor/synth_combined_trans_"$nh"h_"$nt"t \
			--shorthand en_synth_combined_trans_"$nh"h_"$nt"t \
			--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
			--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
			--charlm \
			--charlm_shorthand 1billion \
			--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
			--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
			--tensorboard --transformer --no_transfer --max_steps 20000 \
			--num_trans_heads $nh --num_trans $nt \
			--lr 0.3 --patience 4 
	done
done


# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 	--mode train \
# 	--save_dir ./models/data_extractor/synth_combined_vanilla \
# 	--shorthand en_synth_combined_vanilla \
# 	--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
# 	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 	--tensorboard

# # train from BERT
# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/ner_tagger.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 	--mode train \
# 	--save_dir ./models/data_extractor/synth_combined_vanilla_from_bert \
# 	--shorthand en_synth_combined_vanilla_from_bert \
# 	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 	--tensorboard
# 
# # train only class layers
# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 	--mode train \
# 	--train_classifier_only \
# 	--output_transform \
# 	--save_dir ./models/data_extractor/synth_combined_vanilla_clf_only_outtrans \
# 	--shorthand en_synth_combined_vanilla_clf_only_outtrans \
# 	--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
# 	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 	--tensorboard
# 
# # train only class layers (no output transform)
# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 	--mode train \
# 	--train_classifier_only \
# 	--save_dir ./models/data_extractor/synth_combined_vanilla_clf_only \
# 	--shorthand en_synth_combined_vanilla_clf_only \
# 	--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
# 	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 	--tensorboard

# # train from BERT (with Convolution layer before LSTM) - Joe experiment
# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor/model_w_Conv.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 	--mode train \
# 	--save_dir ./models/data_extractor/synth_combined_vanilla_from_bert \
# 	--shorthand en_synth_combined_vanilla_from_bert \
# 	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 	--tensorboard

echo "WARNING! SYSTEM WILL SHUTDOWN IN 60 SECONDS UNLESS CTRL-C INTERRUPT..."
sleep 60
shutdown now
