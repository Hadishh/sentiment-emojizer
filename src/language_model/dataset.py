
import os
import torch
import json
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class dataset:
    def __init__(self, json_url):
        self.word2idx = {}
        self.idx2word = []
        self.data = None
        with open(json_url, 'r') as f:
            data = json.load(f)
        for key in data.keys():
            sentence = data[key]
            sentence += ["<EOS>"]
            for token in sentence:
                if(token not in self.word2idx):
                    self.idx2word.append(token)
                    self.word2idx[token] = len(self.idx2word) - 1
        corpus = []
        for key in data.keys():
            sentence = data[key] + ['<EOS>']
            words_idx = [self.word2idx[w] for w in sentence]
            sentence = torch.tensor(words_idx).type(torch.int64)
            corpus.append(sentence)
        self.data = torch.cat(corpus)
