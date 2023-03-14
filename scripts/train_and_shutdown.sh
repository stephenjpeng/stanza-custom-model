#!/bin/bash

# train no bilstm
# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 	--mode train \
# 	--save_dir ./models/data_extractor/no_bilstm \
# 	--shorthand no_bilstm \
# 	--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
# 	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 	--no_transfer --no_bilstm \
# 	--tensorboard

# # train trigram_cnn
# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 	--mode train \
# 	--save_dir ./models/data_extractor/nobilstm_trigram_cnn_5L_0.5drop \
# 	--shorthand nobilstm_trigram_cnn_6L_0.5drop \
# 	--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
# 	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 	--trigram_cnn --trigram_drop 0.5 --no_transfer --num_trans 5 --no_bilstm \
# 	--tensorboard

for nh in 16 8
do
	for nt in 2
	do
		sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
			--data_dir ./data \
			--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
			--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
			--mode train \
			--save_dir ./models/data_extractor/trans_"$nh"h_"$nt"t_0.3d_0.2lr_adam \
			--shorthand trans_"$nh"h_"$nt"t_0.3d_0.2lr_adam \
			--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
			--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
			--charlm \
			--charlm_shorthand 1billion \
			--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
			--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
			--tensorboard --transformer --no_transfer \
			--optim adam --batch_size 128 --max_steps 50000 --lr 0.2 --lr_decay 0 --min_lr 0 \
			--num_trans_heads $nh --num_trans $nt --trans_drop 0.3

		sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
			--data_dir ./data \
			--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
			--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
			--mode train \
			--finetune_load_name ./models/data_extractor/trans_"$nh"h_"$nt"t_0.3d_0.2lr_adam/trans_"$nh"h_"$nt"t_0.3d_0.2lr_adam .pt \
			--save_dir ./models/data_extractor/trans_"$nh"h_"$nt"t_0.3d_0.0001lr_adam_finetune \
			--shorthand trans_"$nh"h_"$nt"t_0.3d_0.0001lr_adam_finetune \
			--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
			--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
			--charlm \
			--charlm_shorthand 1billion \
			--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
			--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
			--tensorboard --transformer --no_transfer \
			--optim adam --batch_size 128 --max_steps 50000 --lr 0.0001 --lr_decay 0 --min_lr 0 \
			--num_trans_heads $nh --num_trans $nt --trans_drop 0.3 --finetune
	done
done

# # train trigram_cnn
# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 	--mode train \
# 	--save_dir ./models/data_extractor/vanilla_trigram_cnn_6L_0.5drop \
# 	--shorthand vanilla_trigram_cnn_6L_0.5drop \
# 	--ner_model_file /home/stephen/stanza_resources/en/ner/conll03.pt \
# 	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 	--trigram_cnn --trigram_drop 0.5 --no_transfer --num_trigram 6 \
# 	--tensorboard
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

# # train from BERT (freeze bert)
# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 	--mode train \
# 	--save_dir ./models/data_extractor/vanilla_from_bert_2stage \
# 	--shorthand vanilla_from_bert_2stage \
# 	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 	--bert_model "bert-base-cased" \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 	--no_transfer --freeze_bert \
# 	--tensorboard

# train from BERT (all layers)
# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 	--mode train \
# 	--finetune_load_name ./models/data_extractor/vanilla_from_bert_2stage/vanilla_from_bert_2stage_dataextractor.pt \
# 	--save_dir ./models/data_extractor/vanilla_from_bert_2stage_p2 \
# 	--shorthand vanilla_from_bert_2stage_p2 \
# 	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 	--bert_model "bert-base-cased" \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 	--finetune \
# 	--optim adam --max_steps 50000 --lr 0.00001 --lr_decay 0 --min_lr 0 \
# 	--tensorboard

# # train from BERT no BiLSTM (freeze bert)
# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 	--mode train \
# 	--save_dir ./models/data_extractor/nobilstm_from_bert_2stage \
# 	--shorthand nobilstm_from_bert_2stage \
# 	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 	--bert_model "bert-base-cased" \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 	--no_bilstm --no_transfer --freeze_bert \
# 	--tensorboard

# # train from BERT no BiLSTM (finetune)
# sudo -u stephen PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/data_extractor.py \
# 	--data_dir ./data \
# 	--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 	--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 	--mode train \
# 	--finetune_load_name ./models/data_extractor/nobilstm_from_bert_2stage/nobilstm_from_bert_2stage_dataextractor.pt \
# 	--save_dir ./models/data_extractor/nobilstm_from_bert_2stage_p2 \
# 	--shorthand nobilstm_from_bert_2stage_p2 \
# 	--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 	--bert_model "bert-base-cased" \
# 	--charlm \
# 	--charlm_shorthand 1billion \
# 	--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 	--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 	--no_bilstm --finetune \
# 	--optim adam --max_steps 50000 --lr 0.00001 --lr_decay 0 --min_lr 0 \
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

# # train from pretrained embeddings (with Convolution layer before LSTM) - Joe (AWS)
# sudo -u joe PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/joe_experiment/ner_tagger_joe.py \
# 		--data_dir ./data \
# 		--train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
# 		--eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 		--mode train \
# 	  	--save_dir ./models/joe_experiment/synth_combined_vanilla_from_pretrain_wConv_k3 \
# 	  	--shorthand en_synth_combined_vanilla_from_pretrain_wConv_k3 \
# 		--wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
# 		--charlm \
# 		--charlm_shorthand 1billion \
# 		--charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
# 		--charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
# 		--tensorboard

# # train from BERT add hidden layer --Mitty
# sudo -u mitty PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/mitty_experiment/ner_tagger_mitty.py \
	# --data_dir ./data \
	# --train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
	# --eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
	# --mode train \
	# --save_dir ./models/mitty_experiment/synth_combined_vanilla_from_bert_ablation_l1 \
	# --shorthand en_synth_combined_vanilla_from_bert_ablation_l1 \
	# --wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
	# --charlm \
	# --charlm_shorthand 1billion \
	# --charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
	# --charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
	# --tensorboard


echo "WARNING! SYSTEM WILL SHUTDOWN IN 60 SECONDS UNLESS CTRL-C INTERRUPT..."
sleep 60
shutdown now
