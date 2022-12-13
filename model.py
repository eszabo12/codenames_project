# IMPLEMENT YOUR MODEL CLASS HERE

import numpy as np
import torch
import torch.nn.functional as F
from game import Codenames
from configuration import CodenamesConfiguration
from embeddings.fasttext import FastText
from embeddings.glove import Glove
from game import Codenames

device = torch.device('cpu')

class CodeMaster_Model(torch.nn.Module):
    def __init__(
            self,
            device,
            vocab_size,
            embedding_dim,
            game_state_size,
            args,
            game,
            board
        ):
        super(CodeMaster_Model, self).__init__()
        self.device = device
        self.hidden_size = 128
        self.game_state_size = game_state_size
        self.dropout = 0.01
        self.encoder = torch.nn.LSTM(self.game_state_size, self.hidden_size, dropout=self.dropout)
        self.count = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()
        self.fasttext = FastText()
        self.vectors = self.fasttext.vectors
        self.game = game
        self.board = board
        
    def forward(self, x):
            # returns 5 clues
            count = 2
            best_scores, best_clues, best_board_words_for_clue = self.game.get_clue(count, 1)
            return best_clues[0], count
            embeds = self.vectors(x)
            print("embed", embeds.size())
            out, (hn, cn) = self.lstm(embeds)
            print("lstm", out.size())
            out = torch.transpose(out, dim0=1, dim1=2)
            print("transpose", out.size())
            hint = self.hint(out)
            count = self.count(out)
            return hint, count

    def choose(self, words):
        for word in words:
            self.game.choose(word)

class Guesser_Model(torch.nn.Module):
    def __init__(
            self,
            device,
            vocab_size,
            embedding_dim,
            game_state_size,
            args,
            game,
            tokenizer,
            board
        ):
        super(CodeMaster_Model, self).__init__()
        self.device = device
        self.hidden_size = self.num_outputs
        self.game_state_size = game_state_size
        self.dropout = 0.1
        self.num_actions = 8
        self.n_layers = 1
        self.board_size = 25
        self.glove = Glove()
        self.vectors = self.fasttext.vectors
        self.lstm = torch.nn.Linear(embedding_dim, self.hidden_size, dropout=self.dropout)
        self.chooser = torch.nn.Linear(self.embed_dim, self.board_size)
        self.relu = torch.nn.ReLU()
        self.game = game
        self.board = board

    def forward(self, x):
            embeds = self.vectors[x]
            print("embed", embeds.size())
            out, (hn, cn) = self.lstm(embeds)
            print("lstm", out.size())
            out = torch.transpose(out, dim0=1, dim1=2)
            print("transpose", out.size())
            choice = self.chooser(out)
            value, index = torch.topk(choice, 2)
            for i in index:
                self.game.choose(i)
            return index
