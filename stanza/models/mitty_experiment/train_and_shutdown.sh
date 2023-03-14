#!/bin/bash

# train from BERT add hidden layer and change activation function --Mitty
# sudo -u mitty PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/mitty_experiment/ner_tagger_mitty.py \
#         --data_dir ./data \
#         --train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
#         --eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 		--mode train \
#         --save_dir ./models/mitty_experiment/synth_combined_vanilla_l_gelu \
#         --shorthand en_synth_combined_vanilla_l_gelu \
#         --wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
#         --charlm \
#         --charlm_shorthand 1billion \
#         --charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
#         --charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
#         --add_layer_before_output 1 \
# 		--activation gelu \
#         --tensorboard

# sudo -u mitty PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/mitty_experiment/ner_tagger_mitty.py \
#         --data_dir ./data \
#         --train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
#         --eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 		--mode train \
#         --save_dir ./models/mitty_experiment/synth_combined_vanilla_l_relu \
#         --shorthand en_synth_combined_vanilla_l_relu \
#         --wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
#         --charlm \
#         --charlm_shorthand 1billion \
#         --charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
#         --charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
#         --add_layer_before_output 1 \
# 		--activation relu \
#         --tensorboard

# sudo -u mitty PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/mitty_experiment/ner_tagger_mitty.py \
#         --data_dir ./data \
#         --train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
#         --eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
# 		--mode train \
#         --save_dir ./models/mitty_experiment/synth_combined_vanilla_l_sigmoid \
#         --shorthand en_synth_combined_vanilla_l_sigmoid \
#         --wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
#         --charlm \
#         --charlm_shorthand 1billion \
#         --charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
#         --charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
#         --add_layer_before_output 1 \
# 		--activation sigmoid \
#         --tensorboard

# train from BERT add attention layer and change number of attention head --Mitty
sudo -u mitty PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/mitty_experiment/ner_tagger_mitty.py \
        --data_dir ./data \
        --train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
        --eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
		--mode train \
        --save_dir ./models/mitty_experiment/synth_combined_vanilla_attn_8 \
        --shorthand en_synth_combined_vanilla_attn_8 \
        --wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
        --charlm \
        --charlm_shorthand 1billion \
        --charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
        --charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
        --add_layer_before_output 0 \
        --attn_layer 1 \
        --attn_num_head 8 \
        --tensorboard

sudo -u mitty PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/mitty_experiment/ner_tagger_mitty.py \
        --data_dir ./data \
        --train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
        --eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
		--mode train \
        --save_dir ./models/mitty_experiment/synth_combined_vanilla_attn_16 \
        --shorthand en_synth_combined_vanilla_attn_16 \
        --wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
        --charlm \
        --charlm_shorthand 1billion \
        --charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
        --charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
        --add_layer_before_output 0 \
        --attn_layer 1 \
        --attn_num_head 16 \
        --tensorboard

sudo -u mitty PYTHONPATH=$PYTHONPATH:. /opt/conda/envs/pytorch/bin/python3 stanza/models/mitty_experiment/ner_tagger_mitty.py \
        --data_dir ./data \
        --train_file ./stanza/TOC_Utility/Processed_Data/synth_combined.train.json \
        --eval_file  ./stanza/TOC_Utility/Processed_Data/synth_combined.dev.json \
		--mode train \
        --save_dir ./models/mitty_experiment/synth_combined_vanilla_attn_32 \
        --shorthand en_synth_combined_vanilla_attn_32 \
        --wordvec_pretrain_file /home/stephen/stanza_resources/en/pretrain/combined.pt \
        --charlm \
        --charlm_shorthand 1billion \
        --charlm_forward_file /home/stephen/stanza_resources/en/forward_charlm/1billion.pt \
        --charlm_backward_file /home/stephen/stanza_resources/en/backward_charlm/1billion.pt \
        --add_layer_before_output 0 \
        --attn_layer 1 \
        --attn_num_head 32 \
        --tensorboard

echo "WARNING! SYSTEM WILL SHUTDOWN IN 60 SECONDS UNLESS CTRL-C INTERRUPT..."
sleep 60
shutdown now