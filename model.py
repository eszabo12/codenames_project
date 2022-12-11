# IMPLEMENT YOUR MODEL CLASS HERE

import numpy as np
import torch
import torch.nn.functional as F

device = torch.device('cpu')

class CodeMaster_Model(torch.nn.Module):
    def __init__(
            self,
            device,
            vocab_size,
            embedding_dim,
            game_state_size,
        ):
        super(CodeMaster_Model, self).__init__()
        self.device = device
        self.hidden_size = self.num_outputs
        self.game_state_size = game_state_size
        self.dropout = 0.1
        self.num_actions = 8
        self.n_layers = 1
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=self.padding_idx)
        self.lstm = torch.nn.Linear(embedding_dim, self.hidden_size, dropout=self.dropout)
        self.hint = torch.nn.Linear(self.hidden_size, self.vocab_size)
        self.count = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
            embeds = self.embedding(x)
            print("embed", embeds.size())
            out, (hn, cn) = self.lstm(embeds)
            print("lstm", out.size())
            out = torch.transpose(out, dim0=1, dim1=2)
            print("transpose", out.size())
            hint = self.hint(out)
            count = self.count(out)
            return hint, count



class Guesser_Model(torch.nn.Module):
    def __init__(
            self,
            device,
            vocab_size,
            embedding_dim,
            game_state_size,
            tokenizer
        ):
        super(CodeMaster_Model, self).__init__()
        self.device = device
        self.hidden_size = self.num_outputs
        self.game_state_size = game_state_size
        self.dropout = 0.1
        self.num_actions = 8
        self.n_layers = 1
        self.board_size = 25
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=self.padding_idx)
        self.lstm = torch.nn.Linear(embedding_dim, self.hidden_size, dropout=self.dropout)
        self.hint = torch.nn.Linear(self.hidden_size, 25)
        self.count = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
            embeds = self.embedding(x)
            print("embed", embeds.size())
            out, (hn, cn) = self.lstm(embeds)
            print("lstm", out.size())
            out = torch.transpose(out, dim0=1, dim1=2)
            print("transpose", out.size())
            hint = self.hint(out)
            count = self.count(out)
            return hint, count