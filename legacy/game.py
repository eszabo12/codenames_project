
import heapq
import itertools
import os
import random
import re
import string
import sys
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
from configuration import CodenamesConfiguration

#predefined constants
sys.path.insert(0, "../")

stopwords = [
    'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'get', 'put',
]

idf_lower_bound = 0.0006
default_single_word_label_scores = (1, 1.1, 1.1, 1.2)

class Codenames(object):

    def __init__(
        self,
        embedding_type="custom",
        configuration=None,
        
    ):
        """
        :param embedding_type: e.g.'word2vec', 'glove', 'fasttext', 'babelnet'
        :param embedding: an embedding object that codenames will use to play

        """
        # Intialize variables
        if configuration != None:
            self.configuration = configuration
        else:
            self.configuration = CodenamesConfiguration()
        print("Codenames Configuration: ", self.configuration.__dict__)
        with open('data/word_to_dict2vec_embeddings', 'rb') as word_to_dict2vec_embeddings_file:
            self.word_to_dict2vec_embeddings = pickle.load(word_to_dict2vec_embeddings_file)
        self.embedding_type = embedding_type
        self.embedding = self._get_embedding_from_type(embedding_type)
        self.weighted_nn = dict()

        self.num_docs, self.word_to_df = self._load_document_frequencies()  # dictionary of word to document frequency

        # Used to get word stems
        self.stemmer = PorterStemmer()

    """
    Codenames game setup
    """

    def _get_embedding_from_type(self, embedding_type):
        """
        :param embedding_type: 'babelnet', 'word2vec', glove', 'fasttext'
        returns the embedding object that will be used to play

        """
        print("Building game for ", embedding_type, "...")
        if embedding_type == 'babelnet':
            return Babelnet(self.configuration)
        elif embedding_type == 'word2vec':
            return Word2Vec(self.configuration)
        elif embedding_type == 'glove':
            return Glove(self.configuration)
        elif embedding_type == 'fasttext':
            print("fasttext chosen")
            return FastText(self.configuration)
        elif embedding_type == 'bert':
            return Bert(self.configuration)
        elif embedding_type == 'kim2019':
            return Kim2019(self.configuration, self.word_to_dict2vec_embeddings)
        else:
            print("Valid embedding types are babelnet, word2vec, glove, fasttext, and bert")

        return None

    def _build_game(self, red=None, blue=None, black=None, save_path=None):
        """
        :param red: optional list of strings of opponent team's words
        :param blue: optional list of strings of our team's words
        :param save_path: optional directory path to save data between games
        :return: None
        """
        self._generate_board_words(red, blue, black)
        self.save_path = save_path
        self.weighted_nn = dict()

        self.words = self.blue_words.union(self.red_words).union(self.black_word)
        for word in self.words:
            self.weighted_nn[word] = self.embedding.get_weighted_nn(word)

        self._write_to_debug_file(["\n", "Building game with configuration:", self.configuration.description(), "\n\tBLUE words: ", " ".join(self.blue_words), "RED words:", " ".join(self.red_words), "\n"])

    def _generate_board_words(self, red=None, blue=None, black=None):
        """
        :param red: optional list of strings of opponent team's words
        :param blue: optional list of strings of our team's words
        :return: None
        """
        idx_to_word = dict()

        with open("data/codewords.txt") as file:
            for i, line in enumerate(file):
                word = line.strip().lower()
                idx_to_word[i] = word

        rand_idxs = random.sample(range(0, len(idx_to_word.keys())), 10)

        self.red_words = set([idx_to_word[idx] for idx in rand_idxs[:5]])
        self.blue_words = set([idx_to_word[idx] for idx in rand_idxs[5:]])

        if red is not None:
            self.red_words = set(red)
        if blue is not None:
            self.blue_words = set(blue)
        if black is not None:
            self.black_word = set(black)

    def _load_document_frequencies(self):
        """
        Sets up a dictionary from words to their document frequency
        """
        if (os.path.exists("data/word_to_df.pkl")) and (os.path.exists("data/text8_num_documents.txt")):
            with open('data/word_to_df.pkl', 'rb') as f:
                word_to_df = pickle.load(f)
            with open('data/text8_num_documents.txt', 'rb') as f:
                for line in f:
                    num_docs = int(line.strip())
                    break
        else:
            dataset = api.load("text8")
            dct = Dictionary(dataset)
            id_to_doc_freqs = dct.dfs
            num_docs = dct.num_docs
            word_to_df = {dct[id]: id_to_doc_freqs[id]
                          for id in id_to_doc_freqs}

        return num_docs, word_to_df


    def _write_to_debug_file(self, lst):
        if self.configuration.debug_file:
            with open(self.configuration.debug_file, 'a') as f:
                f.write(" ".join([str(x) for x in lst]))

    '''
    Codenames game methods
    '''
    #2,1
    def get_clue(self, n, penalty):
        # where blue words are our team's words and red words are the other team's words
        # potential clue candidates are the intersection of weighted_nns[word] for each word in blue_words
        # we need to repeat this for the (|blue_words| C n) possible words we can give a clue for

        pq = []
        for word_set in itertools.combinations(self.blue_words, n):
            highest_clues, score = self.get_highest_clue(
                word_set, penalty)
            # min heap, so push negative score
            heapq.heappush(pq, (-1 * score, highest_clues, word_set))

        # sliced_labels = self.get_cached_labels_from_synset(clue)
        # main_sense, _senses = self.get_cached_labels_from_synset_v5(clue)

        best_clues = []
        best_board_words_for_clue = []
        best_scores = []
        count = 0

        while pq:
            score, clues, word_set = heapq.heappop(pq)

            if count >= 5:
                break

            if self.configuration.visualize and callable(getattr(self.embedding, "get_intersecting_graphs", None)):
                for clue in clues:
                    self.embedding.get_intersecting_graphs(
                        word_set,
                        clue,
                        split_multi_word=self.configuration.split_multi_word,
                    )

            best_clues.append(clues)
            best_scores.append(score)
            best_board_words_for_clue.append(word_set)

            count += 1

        return best_scores, best_clues, best_board_words_for_clue

    def is_valid_clue(self, clue):
        # no need to remove red/blue words from potential_clues elsewhere
        # since we check for validity here
        for board_word in self.words:
            # Check if clue or board_word are substring of each other, or if they share the same word stem
            if (clue in board_word or board_word in clue or self.stemmer.stem(clue) == self.stemmer.stem(board_word) or not clue.isalpha()):
                return False
        return True

    def get_highest_clue(self, chosen_words, penalty=1.0):

        if self.embedding_type == 'kim2019':
            chosen_clue, dist = self.embedding.get_clue(
                self.blue_words, self.red_words, chosen_words)
            # return the angular similarity
            return [chosen_clue], 1 - dist

        potential_clues = set()
        for word in chosen_words:
            nns = self.weighted_nn[word]
            potential_clues.update(nns)

        highest_scoring_clues = []
        highest_score = float("-inf")

        for clue in potential_clues:
            # don't consider clues which are a substring of any board words
            if not self.is_valid_clue(clue):
                continue
            blue_word_counts = []
            for blue_word in chosen_words:
                if clue in self.weighted_nn[blue_word]:
                    blue_word_counts.append(self.weighted_nn[blue_word][clue])
                else:
                    blue_word_counts.append(self.embedding.get_word_similarity(blue_word, clue))

            heuristic_score = 0

            self._write_to_debug_file([
                "\n", clue, "score breakdown for", " ".join(chosen_words),
                "\n\tblue words score:", round(sum(blue_word_counts),3),
            ])

            if self.configuration.use_heuristics is True:
                # the larger the idf is, the more uncommon the word
                idf = (1.0/self.word_to_df[clue]) if clue in self.word_to_df else 1.0

                # prune out super common words (e.g. "get", "go")
                if (clue in stopwords or idf < idf_lower_bound):
                    idf = 1.0
                dict2vec_weight = self.embedding.dict2vec_embedding_weight()
                dict2vec_score = dict2vec_weight*get_dict2vec_score(chosen_words, clue, self.red_words)
                black_score = dict2vec_weight*get_dict2vec_score(chosen_words, clue, self.black_word)
                heuristic_score = dict2vec_score + black_score + (-2*idf)
                self._write_to_debug_file([" IDF:", round(-2*idf,3), "dict2vec score:", round(dict2vec_score,3)])

            # Give embedding methods the opportunity to rescale the score using their own heuristics
            embedding_score = self.embedding.rescale_score(chosen_words, clue, self.red_words, self.black_word)

            if (self.configuration.use_kim_scoring_function):
                score = min(blue_word_counts) + heuristic_score
            else:
                score = sum(blue_word_counts) + embedding_score + heuristic_score

            if score > highest_score:
                highest_scoring_clues = [clue]
                highest_score = score
            elif score == highest_score:
                highest_scoring_clues.append(clue)

        return highest_scoring_clues, highest_score

    def choose_words(self, n, clue, remaining_words):
        # given a clue word, choose the n words from remaining_words that most relates to the clue

        pq = []

        for word in remaining_words:
            score = self.get_score(clue, word)
            # min heap, so push negative score
            heapq.heappush(pq, (-1 * score, word))

        ret = []
        for i in range(n):
            ret.append(heapq.heappop(pq))
        return ret

    def get_score(self, clue, word):
        """
        :param clue: string
        :param possible_words: n-tuple of strings
        :return: score = sum(weighted_nn[possible_word][clue] for possible_word in possible_words)
        """
        if clue in self.weighted_nn[word]:
            return self.weighted_nn[word][clue]
        else:
            try:
                return self.embedding.get_word_similarity(word, clue)
            except KeyError:
                return -1000