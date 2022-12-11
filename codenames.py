
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

# Embeddings
from embeddings.babelnet import Babelnet
from embeddings.word2vec import Word2Vec
from embeddings.glove import Glove
from embeddings.fasttext import FastText
from embeddings.bert import Bert
from embeddings.kim2019 import Kim2019

from utils import get_dict2vec_score
from board import Board
from game import Codenames
from configuration import CodenamesConfiguration



# #predefined constants
# sys.path.insert(0, "../")

# stopwords = [
#     'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'get', 'put',
# ]

# idf_lower_bound = 0.0006
default_single_word_label_scores = (1, 1.1, 1.1, 1.2)


if __name__ == "__main__":
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
    args = parser.parse_args()


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
    print("len words", len(words))

    red_words = []
    blue_words = []
    black_words = []

    for _ in range(0, args.num_trials):
        random.shuffle(words)
        red_words.append(words[:12])
        blue_words.append(words[12:24])
        black_words.append([words[24]])

    amt_file_path = 'amt_102620_all_kim_scoring_fx.csv'
    amt_key_file_path = 'amt_102620_all_kim_scoring_fx_key.csv'
    # Setup CSVs
    if not os.path.exists(amt_file_path):
        with open(amt_file_path, 'w'): pass

    with open(amt_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header_row = ['embedding_name', 'clue'] + ["word" + str(x) for x in range(0,20)]
        writer.writerow(header_row)

    if not os.path.exists(amt_key_file_path):
        with open(amt_key_file_path, 'w'): pass

    with open(amt_key_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header_row = ['embedding_name', 'configuration','clue', 'word0ForClue', 'word1ForClue'] + ["blueWord" + str(x) for x in range(0,10)] + ["redWord" + str(x) for x in range(0,10)]
        writer.writerow(header_row)

    embedding_trial_to_clues = dict()

    shuffled_embeddings = args.embeddings
    random.shuffle(shuffled_embeddings)

    for embedding_type in shuffled_embeddings:
        embedding_trial_number = 0
        debug_file_path = None
        if args.debug is True or args.debug_file != None:
            debug_file_path = (embedding_type + "-" + datetime.now().strftime("%m-%d-%Y-%H.%M.%S") + ".txt") if args.debug_file == None else args.debug_file
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
        
        game = Codenames(
            configuration=configuration,
            embedding_type=embedding_type
        )
        
        
        for i, (red, blue, black) in enumerate(zip(red_words, blue_words, black_words)):
            game._build_game(red=red, blue=blue, black=black,
                             save_path="tmp_babelnet_" + str(i))
            board = Board(red, blue, black)
            if game.configuration.verbose:
                print("NEAREST NEIGHBORS:")
                for word, clues in game.weighted_nn.items():
                    print(word)
                    print(sorted(clues, key=lambda k: clues[k], reverse=True)[:5])

            best_scores, best_clues, best_board_words_for_clue = game.get_clue(2, 1)

            print("==================================================================================================================")
            print("TRIAL", str(i+1))
            board.print()
            print("BEST CLUES: ")
            for score, clues, board_words in zip(best_scores, best_clues, best_board_words_for_clue):
                print()
                print("Clue(s):", ", ".join(clues), "|| Intended board words:", board_words, "|| Score:", str(round(score,3)))

            # Write to CSV
            heuristic_string = "WithHeuristics" if configuration.use_heuristics else "WithoutHeuristics"
            kim_scoring_fx_string = "KimFx" if configuration.use_kim_scoring_function else "WithoutKimFx"
            embedding_with_trial_number = embedding_type +  heuristic_string + kim_scoring_fx_string + "Trial" + str(embedding_trial_number)

            # Check if this clue has already been chosen
            embedding_number = embedding_type + str(embedding_trial_number)
            clue = best_clues[0][0]
            is_duplicate_clue = embedding_number in embedding_trial_to_clues and clue in embedding_trial_to_clues[embedding_number]
            if (is_duplicate_clue is False):
                if embedding_number not in embedding_trial_to_clues:
                    embedding_trial_to_clues[embedding_number] = set()
                embedding_trial_to_clues[embedding_number].add(clue)

                with open(amt_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([embedding_with_trial_number, clue] + list(game.words))

            with open(amt_key_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([embedding_with_trial_number, str(configuration.__dict__), clue, best_board_words_for_clue[0][0], best_board_words_for_clue[0][1]] + list(game.blue_words) + list(game.red_words))

            embedding_trial_number += 1
