

import torch
import heapq
import itertools
import os
import random
import re
import string
import urllib
import string
import argparse
import gzip
import pickle
import math
from datetime import datetime
import csv
import gym
# Gensim
from gensim.corpora import Dictionary
import gensim.downloader as api

# nltk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer

import networkx as nx
import numpy as np
import requests
from tqdm import tqdm

from utils import get_dict2vec_score
from board import Board
from configuration import CodenamesConfiguration

from env import CodemasterEnv, GuesserEnv

from embeddings.fasttext import FastText

from model import CodeMaster_Model, Guesser_Model


words = [
        'vacuum', 'whip', 'moon', 'school', 'tube', 'lab', 'key', 'table', 'lead', 'crown',
        'bomb', 'bug', 'pipe', 'roulette', 'play', 'cloak', 'piano', 'beijing', 'bison',
        'boot', 'cap', 'car','change', 'circle', 'cliff', 'conductor', 'cricket', 'death', 'diamond',
        'figure', 'gas', 'germany', 'india', 'jupiter', 'kid', 'king', 'lemon', 'litter', 'nut',
        'phoenix', 'racket', 'row', 'scientist', 'shark', 'stream', 'swing', 'unicorn', 'witch', 'worm',
        'pistol', 'saturn', 'rock', 'superhero', 'mug', 'fighter', 'embassy', 'cell', 'state', 'beach',
        'capital', 'post', 'cast', 'soul', 'tower', 'green', 'plot', 'string', 'kangaroo', 'lawyer', 'fire',
        'robot', 'mammoth', 'hole', 'spider', 'bill', 'ivory', 'giant', 'bar', 'ray', 'drill', 'staff',
        'greece', 'press','pitch', 'nurse', 'contract', 'water', 'watch', 'amazon','spell', 'kiwi', 'ghost',
        'cold', 'doctor', 'port', 'bark','foot', 'luck', 'nail', 'ice', 'needle', 'disease', 'comic', 'pool',
        'field', 'star', 'cycle', 'shadow', 'fan', 'compound', 'heart', 'flute','millionaire', 'pyramid', 'africa',
        'robin', 'chest', 'casino','fish', 'oil', 'alps', 'brush', 'march', 'mint','dance', 'snowman', 'torch',
        'round', 'wake', 'satellite','calf', 'head', 'ground', 'club', 'ruler', 'tie','parachute', 'board',
        'paste', 'lock', 'knight', 'pit', 'fork', 'egypt', 'whale', 'scale', 'knife', 'plate','scorpion', 'bottle',
        'boom', 'bolt', 'fall', 'draft', 'hotel', 'game', 'mount', 'train', 'air', 'turkey', 'root', 'charge',
        'space', 'cat', 'olive', 'mouse', 'ham', 'washer', 'pound', 'fly', 'server','shop', 'engine',
        'box', 'shoe', 'tap', 'cross', 'rose', 'belt', 'thumb', 'gold', 'point', 'opera', 'pirate',
        'tag', 'olympus', 'cotton', 'glove', 'sink', 'carrot', 'jack', 'suit', 'glass', 'spot', 'straw', 'well',
        'pan', 'octopus', 'smuggler', 'grass', 'dwarf', 'hood', 'duck', 'jet', 'mercury',
    ]

def reset(args):
    red_words = []
    blue_words = []
    black_words = []

    for _ in range(0, args.num_trials):
        random.shuffle(words)
        red_words.append(words[:12])
        blue_words.append(words[12:24])
        black_words.append([words[24]])
    return red_words, blue_words, black_words
    
    
def create_tokenizer(words):
    tokenizer = {word: i for word, i in enumerate(words)}
    return tokenizer

if __name__ == "__main__":
    default_single_word_label_scores = (1, 1.1, 1.1, 1.2)
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('embeddings', nargs='+',
                            help='an embedding method to use when playing codenames')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='print out verbose information'),
    parser.add_argument('--visualize', dest='visualize', action='store_true',
                        help='visualize the choice of clues with graphs')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Write score breakdown to a file. You can specify what file is used with --debug-file, or one will be created for you')
    parser.add_argument('--no-heuristics', dest='no_heuristics', action='store_true',
                        help='Remove heuristics such as IDF and dict2vec')
    parser.add_argument('--debug-file', dest='debug_file', default=None,
                        help='Write score breakdown to debug file')
    parser.add_argument('--num-trials', type=int, dest='num_trials', default=1,
                        help='number of trials of the game to run')
    parser.add_argument('--split-multi-word', dest='split_multi_word', default=True)
    parser.add_argument('--disable-verb-split', dest='disable_verb_split', default=True)
    parser.add_argument('--kim-scoring-function', dest='use_kim_scoring_function', action='store_true',
                        help='use the kim 2019 et. al. scoring function'),
    parser.add_argument('--length-exp-scaling', type=int, dest='length_exp_scaling', default=None,
                        help='Rescale lengths using exponent')
    parser.add_argument('--single-word-label-scores', type=float, nargs=4, dest='single_word_label_scores',
                        default=default_single_word_label_scores,
                        help='main_single, main_multi, other_single, other_multi scores')
    parser.add_argument('--babelnet-api-key', type=str, dest='babelnet_api_key', default=None)
    parser.add_argument('--words-per-clue', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=25)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    args = parser.parse_args()
    debug_file_path = None
    if args.debug is True or args.debug_file != None:
        debug_file_path = (args.embeddings[0] + "-" + datetime.now().strftime("%m-%d-%Y-%H.%M.%S") + ".txt") if args.debug_file == None else args.debug_file
        # Create directory to put debug files if it doesn't exist
        if not os.path.exists('debug_output'):
            os.makedirs('debug_output')
        debug_file_path = os.path.join('debug_output', debug_file_path)
        print("Writing debug output to", debug_file_path)
    configuration = CodenamesConfiguration(
                verbose=args.verbose,
                visualize=args.visualize,
                split_multi_word=args.split_multi_word,
                disable_verb_split=args.disable_verb_split,
                debug_file=debug_file_path,
                length_exp_scaling=args.length_exp_scaling,
                use_heuristics=(not args.no_heuristics),
                single_word_label_scores=args.single_word_label_scores,
                use_kim_scoring_function=args.use_kim_scoring_function,
                babelnet_api_key=args.babelnet_api_key,
            )
    if args.debug is True or args.debug_file != None:
        debug_file_path = (args.embeddings[0] + "-" + datetime.now().strftime("%m-%d-%Y-%H.%M.%S") + ".txt") if args.debug_file == None else args.debug_file
        # Create directory to put debug files if it doesn't exist
        if not os.path.exists('debug_output'):
            os.makedirs('debug_output')
        debug_file_path = os.path.join('debug_output', debug_file_path)
        print("Writing debug output to", debug_file_path)

    tokenizer = create_tokenizer(words)
    vocab_size = len(tokenizer)
    red_words, blue_words, black_words = reset(args)
    game_state_size = 3
    codemaster_model = CodeMaster_Model(torch.device("cpu"), vocab_size, 300, game_state_size, args)
    guesser_model = Guesser_Model
    for trial in range(args.num_trials):
        print("TRIAL", str(trial+1))
        board = Board(red_words[trial], blue_words[trial], black_words[trial])
        print(board.get_state().size())
        for step in range(args.max_steps):
            print("STEP", str(step+1))
            board.print()
            # first the codemaster needs to give a hint
            # codemaster_model
            #then the guesser model needs to guess