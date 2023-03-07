import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, pad_sequence, PackedSequence
from stanza.models.common.data import map_to_ids, get_long_tensor

from stanza.models.common.trigram_cnn import TrigramCNN
from stanza.models.common.dropout import WordDropout, LockedDropout
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from stanza.models.common.crf import CRFLoss
from stanza.models.common.vocab import PAD_ID, UNK_ID
from stanza.models.common.bert_embedding import extract_bert_embeddings
logger = logging.getLogger('stanza')

class TransitionDataExtractor(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None, bert_model=None, bert_tokenizer=None, use_cuda=False):
        super().__init__()

        self.use_cuda = use_cuda
        self.vocab = vocab
        self.args = args
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # this will remember None if there is no bert
        add_unsaved_module('bert_model', bert_model)
        add_unsaved_module('bert_tokenizer', bert_tokenizer)

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            emb_finetune = self.args.get('emb_finetune', True)

            # load pretrained embeddings if specified
            word_emb = nn.Embedding(len(self.vocab['word']), self.args['word_emb_dim'], PAD_ID)
            # if a model trained with no 'delta' vocab is loaded, and
            # emb_finetune is off, any resaving of the model will need
            # the updated vectors.  this is accounted for in load()
            if not emb_finetune or 'delta' in self.vocab:
                # if emb_finetune is off
                # or if the delta embedding is present
                # then we won't fine tune the original embedding
                add_unsaved_module('word_emb', word_emb)
                self.word_emb.weight.detach_()
            else:
                self.word_emb = word_emb
            if emb_matrix is not None:
                self.init_emb(emb_matrix)

            # TODO: allow for expansion of delta embedding if new
            # training data has new words in it?
            self.delta_emb = None
            if 'delta' in self.vocab:
                # zero inits seems to work better
                # note that the gradient will flow to the bottom and then adjust the 0 weights
                # as opposed to a 0 matrix cutting off the gradient if higher up in the model
                self.delta_emb = nn.Embedding(len(self.vocab['delta']), self.args['word_emb_dim'], PAD_ID)
                nn.init.zeros_(self.delta_emb.weight)
                # if the model was trained with a delta embedding, but emb_finetune is off now,
                # then we will detach the delta embedding
                if not emb_finetune:
                    self.delta_emb.weight.detach_()

            input_size += self.args['word_emb_dim']

        if self.bert_model is not None:
            input_size += self.bert_model.config.hidden_size

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args['charlm']:
                if args['charlm_forward_file'] is None or not os.path.exists(args['charlm_forward_file']):
                    raise FileNotFoundError('Could not find forward character model: {}  Please specify with --charlm_forward_file'.format(args['charlm_forward_file']))
                if args['charlm_backward_file'] is None or not os.path.exists(args['charlm_backward_file']):
                    raise FileNotFoundError('Could not find backward character model: {}  Please specify with --charlm_backward_file'.format(args['charlm_backward_file']))
                add_unsaved_module('charmodel_forward', CharacterLanguageModel.load(args['charlm_forward_file'], finetune=False))
                add_unsaved_module('charmodel_backward', CharacterLanguageModel.load(args['charlm_backward_file'], finetune=False))
                input_size += self.charmodel_forward.hidden_dim() + self.charmodel_backward.hidden_dim()
            else:
                self.charmodel = CharacterModel(args, vocab, bidirectional=True, attention=False)
                input_size += self.args['char_hidden_dim'] * 2

        # optionally add a input transformation layer
        if self.args.get('input_transform', False):
            self.input_transform = nn.Linear(input_size, input_size)
        else:
            self.input_transform = None
       
        # trigram CNN layers
        self.tcnn1 = TrigramCNN(input_size, input_size * 5)
        self.tcnn2 = TrigramCNN(input_size, input_size * 5)
        self.tcnn3 = TrigramCNN(input_size, input_size * 5)
        self.tcnn4 = TrigramCNN(input_size, input_size * 5)

        # action classifier
        num_actions = 2 + len(set([i[2:] for i in self.vocab['tag']['_id2unit'] if i[:2] in ('B-', 'I-', 'E-', 'S-')]))
        self.mlp1 = nn.Linear(input_size * 6, self.args['hidden_dims'])
        self.relu1 = nn.ReLU()
        self.mlp2 = nn.Linear(self.args['hidden_dims'], self.args['hidden_dims'])
        self.relu2 = nn.ReLU()
        self.tag_clf = nn.Linear(self.args['hidden_dim'], num_actions)
        self.tag_clf.bias.data.zero_()

        # criterion
        self.crit = nn.CrossEntropyLoss(num_tag)

        self.drop = nn.Dropout(args['dropout'])
        # self.worddrop = WordDropout(args['word_dropout'])
        # self.lockeddrop = LockedDropout(args['locked_dropout'])

    def init_emb(self, emb_matrix):
        if isinstance(emb_matrix, np.ndarray):
            emb_matrix = torch.from_numpy(emb_matrix)
        vocab_size = len(self.vocab['word'])
        dim = self.args['word_emb_dim']
        assert emb_matrix.size() == (vocab_size, dim), \
            "Input embedding matrix must match size: {} x {}, found {}".format(vocab_size, dim, emb_matrix.size())
        self.word_emb.weight.data.copy_(emb_matrix)

    def forward(self, sentences, wordchars, wordchars_mask, tags, word_orig_idx,
            sentlens, wordlens, chars, charoffsets, charlens, char_orig_idx,
            step_nums):
        
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)
        
        inputs = []
        batch_size = len(sentences)

        if self.args['word_emb_dim'] > 0:
            #extract static embeddings
            static_words, word_mask = self.extract_static_embeddings(self.args, sentences, self.vocab['word'])

            if self.use_cuda:
                word_mask = word_mask.cuda()
                static_words = static_words.cuda()
                
            word_static_emb = self.word_emb(static_words)

            if 'delta' in self.vocab and self.delta_emb is not None:
                # masks should be the same
                delta_words, _ = self.extract_static_embeddings(self.args, sentences, self.vocab['delta'])
                if self.use_cuda:
                    delta_words = delta_words.cuda()
                # unclear whether to treat words in the main embedding
                # but not in delta as unknown
                # simple heuristic though - treating them as not
                # unknown keeps existing models the same when
                # separating models into the base WV and delta WV
                # also, note that at training time, words like this
                # did not show up in the training data, but are
                # not exactly UNK, so it makes sense
                delta_unk_mask = torch.eq(delta_words, UNK_ID)
                static_unk_mask = torch.not_equal(static_words, UNK_ID)
                unk_mask = delta_unk_mask * static_unk_mask
                delta_words[unk_mask] = PAD_ID

                delta_emb = self.delta_emb(delta_words)
                word_static_emb = word_static_emb + delta_emb

            word_emb = pack(word_static_emb)
            inputs += [word_emb]

        if self.bert_model is not None:
            device = next(self.parameters()).device
            processed_bert = extract_bert_embeddings(self.args['bert_model'], self.bert_tokenizer, self.bert_model, sentences, device, keep_endpoints=False)
            processed_bert = pad_sequence(processed_bert, batch_first=True)
            inputs += [pack(processed_bert)]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, word_emb.batch_sizes), batch_first=True)[0]

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args.get('charlm', None):
                char_reps_forward = self.charmodel_forward.get_representation(chars[0], charoffsets[0], charlens, char_orig_idx)
                char_reps_forward = PackedSequence(char_reps_forward.data, char_reps_forward.batch_sizes)
                char_reps_backward = self.charmodel_backward.get_representation(chars[1], charoffsets[1], charlens, char_orig_idx)
                char_reps_backward = PackedSequence(char_reps_backward.data, char_reps_backward.batch_sizes)
                inputs += [char_reps_forward, char_reps_backward]
            else:
                char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
                char_reps = PackedSequence(char_reps.data, char_reps.batch_sizes)
                inputs += [char_reps]

        tcnn_inputs = pad(torch.cat([x.data for x in inputs], 1))
        tcnn_inputs = self.tcnn1(tcnn_inputs)
        tcnn_inputs = self.tcnn2(tcnn_inputs)
        tcnn_inputs = self.tcnn3(tcnn_inputs)
        tcnn_inputs = self.tcnn4(tcnn_inputs)

        # prediction layer
        out = self.drop(tcnn_inputs)
        out = self.relu1(self.mlp1(out))
        out = self.relu2(self.mlp2(out))
        logits = self.tag_clf(out)

        gold_actions = 
        loss, trans = self.crit(logits, gold_actions)

        return loss, logits, trans


    @staticmethod
    def extract_static_embeddings(args, sents, vocab):
        processed = []
        if args.get('lowercase', True): # handle word case
            case = lambda x: x.lower()
        else:
            case = lambda x: x
        for idx, sent in enumerate(sents):
            processed_sent = [vocab.map([case(w) for w in sent])]
            processed.append(processed_sent[0])

        words = get_long_tensor(processed, len(sents))
        words_mask = torch.eq(words, PAD_ID)

        return words, words_mask

    @staticmethod
    def extract_gold_actions(sents, tags, steps):
        # TODO: FIXME
        self.vocab['tag']
        num_actions = 2 + len(set([i[2:] for i in self.vocab['tag']['_id2unit'] if i[:2] in ('B-', 'I-', 'E-', 'S-')]))
