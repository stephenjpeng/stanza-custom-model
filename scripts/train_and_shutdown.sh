#!/bin/bash

for nh in 8 16
do
	for nt in 6
	do
		sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
			--data_dir ./data \
			--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
			--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
			--mode train \
			--save_dir ./models/data_extractor/trans_"$nh"h_"$nt"t_0.3d_adam \
			--shorthand trans_"$nh"h_"$nt"t_0.3d_adam \
			--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
			--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
			--charlm \
			--charlm_shorthand 1billion \
			--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
			--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
			--tensorboard --transformer --no_transfer \
			--optim adam --batch_size 256 --max_steps 30000 --lr 0.001 --min_lr 0 \
			--num_trans_heads $nh --num_trans $nt --trans_drop 0.3
	done
done

# train trigram_cnn
sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
	--data_dir ./data \
	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
	--mode train \
	--save_dir ./models/data_extractor/vanilla_trigram_cnn_6L_0.5drop \
	--shorthand vanilla_trigram_cnn_6L_0.5drop \
	--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
	--charlm \
	--charlm_shorthand 1billion \
	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
	--trigram_cnn --trigram_drop 0.5 --no_transfer --num_trans 6 \
	--tensorboard
# 
# 
# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 	--mode train \
# 	--save_dir ./models/data_extractor/synth_combined_trans_16h_4t \
# 	--shorthand en_synth_combined_trans_16h_4t \
# 	--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
# 	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 	--tensorboard --transformer --no_transfer --trans_dropout 0.5 \
# 	--num_trans_heads 16 --num_trans 4

# train transformer (with grid search)


# for nh in 4 8 16
# do
# 	for nt in 4 6 8
# 	do
# 		sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
# 			--data_dir ./data \
# 			--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 			--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 			--mode train \
# 			--save_dir ./models/data_extractor/synth_combined_trans_"$nh"h_"$nt"t \
# 			--shorthand en_synth_combined_trans_"$nh"h_"$nt"t \
# 			--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
# 			--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 			--charlm \
# 			--charlm_shorthand 1billion \
# 			--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 			--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 			--tensorboard --transformer --no_transfer --max_steps 20000 \
# 			--num_trans_heads $nh --num_trans $nt \
# 			--lr 0.3 --patience 4 
# 	done
# done


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

# # train from BERT (with Convolution layer before LSTM) - Joe (local)
	# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/joe_experiment/ner_tagger_joe.py \
	# PYTHONPATH=$PYTHONPATH:. python3 stanza/models/joe_experiment/ner_tagger_joe.py \
	# --data_dir ./data \
	# --train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
	# --eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
	# --mode train \
	# --save_dir ./models/joe_experiment/synth_combined_vanilla_from_bert_wConv \
	# --shorthand en_synth_combined_vanilla_from_bert_wConv \
	# --wordvec_pretrain_file ../stanza_resources/en/pretrain/combined.pt \
	# --charlm \
	# --charlm_shorthand 1billion \
	# --charlm_forward_file ../stanza_resources/en/forward_charlm/1billion.pt \
	# --charlm_backward_file ../stanza_resources/en/backward_charlm/1billion.pt \
	# --tensorboard

# # train from BERT add hidden layer --Mitty
# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/ner_tagger.py \
# PYTHONPATH=$PYTHONPATH:. python3 stanza/models/mitty_experiment/ner_tagger_mitty.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 	--mode train \
# 	--save_dir ./models/mitty_experiment/synth_combined_vanilla_from_bert_ablation_l1 \
# 	--shorthand en_synth_combined_vanilla_from_bert_ablation_l1 \
# 	--wordvec_pretrain_file ../stanza_resources/en/pretrain/combined.pt \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file ../stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file ../stanza_resources/en/backward_charlm/1billion.pt \
# 	--tensorboard


echo "WARNING! SYSTEM WILL SHUTDOWN IN 60 SECONDS UNLESS CTRL-C INTERRUPT..."
sleep 60
shutdown now
