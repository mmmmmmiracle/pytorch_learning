import sys, os
import collections
import zipfile
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch import optim


class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = collections.Counter(tokens)
        self.token_freqs = list(counter.items())
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['', '', '', '']
        else:
            self.unk = 0
            self.idx_to_token += ['']
        self.idx_to_token += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in self.idx_to_token]
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

class TextPreprocessor():
    def __init__(self, text, num_lines):
        self.num_lines = num_lines
        text = self.clean_raw_text(text)
        self.src_tokens, self.tar_tokens = self.tokenize(text)
        self.src_vocab = self.build_vocab(self.src_tokens)
        self.tar_vocab = self.build_vocab(self.tar_tokens)
    
    def clean_raw_text(self, text):
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        out = ''
        for i, char in enumerate(text.lower()):
            if char in (',', '!', '.') and i > 0 and text[i-1] != ' ':
                out += ' '
            out += char
        return out
        
    def tokenize(self, text):
        sources, targets = [], []
        for i, line in enumerate(text.split('\n')):
            if i > self.num_lines:
                break
            parts = line.split('\t')
            if len(parts) >= 2:
                sources.append(parts[0].split(' '))
                targets.append(parts[1].split(' '))
        return sources, targets
        
    def build_vocab(self, tokens):
        tokens = [token for line in tokens for token in line]
        return Vocab(tokens, min_freq=3, use_special_tokens=True)
    
class TextUtil():
    def __init__(self, tp, max_len):
        self.src_vocab, self.tar_vocab = tp.src_vocab, tp.tar_vocab
        src_arr, src_valid_len = self.build_array(tp.src_tokens, tp.src_vocab, max_len = max_len, padding_token = tp.src_vocab.pad, is_source=True)
        tar_arr, tar_valid_len = self.build_array(tp.tar_tokens, tp.tar_vocab, max_len = max_len, padding_token = tp.tar_vocab.pad, is_source=False)
        self.dataset = torch.utils.data.TensorDataset(src_arr, src_valid_len, tar_arr, tar_valid_len)
        
    def build_array(self,lines, vocab, max_len, padding_token, is_source):
        def _pad(line):
            if len(line) > max_len:
                return line[:max_len]
            else:
                return line + (max_len - len(line)) * [padding_token]
        lines = [vocab[line] for line in lines]
        if not is_source:
            lines = [[vocab.bos] + line + [vocab.eos] for line in lines]
        arr = torch.tensor([_pad(line) for line in lines])
        valid_len = (arr != vocab.pad).sum(1)
        return arr, valid_len
        
    def load_data_nmt(self, batch_size):
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size, shuffle = True)
        return self.src_vocab, self.tar_vocab, train_loader