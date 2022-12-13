import numpy as np
import env
import csv
import os
import datetime
import gym
from gym import spaces
import random
import time
from configuration import CodenamesConfiguration
from board import Board
from gym.envs.registration import register

class CodemasterEnv(gym.Env):
    def __init__(self, words, tokenizer, args):
            num_words = len(words)
            self.possible_agents = ["master"]
            self.words_per_clue = args.words_per_clue
            self.tokenizer = tokenizer
            self.action_space = spaces.Discrete(num_words)
            self.observation_space = spaces.Dict(
                {
                    "words": spaces.Box(0, num_words, shape=(25,), dtype=int),
                    "colors": spaces.Box(0, 2, shape=(25,), dtype=int),
                    "chosen": spaces.Box(0, 1, shape=(25,), dtype=int),
                }
            )
            self.board = None
            self.game = None
            self.words = words
            self.args = args
            self.reset()
    def step(self, action):
            # map it to words
            done = False
            if action not in self.words:
                reward = -1
            else:
                word = self.words[action]
                result = self.board.choose(word)
                if self.board.get_color(word) == "black":
                    done = True
                    reward = -2
                else:
                    reward = -1 if self.board.get_color(word) == "red" else 0
            info = {}
            state = self.get_state()
            return state, reward, done, info

    #get the state from the board state
    def observe(self):
        return self.board.get_state()
    def display(self):
        self.board.print()
    def get_action(self):
        best_scores, best_clues, best_board_words_for_clue = self.game.get_clue(2, 1)
    def reset(self):
        red_words = []
        blue_words = []
        black_words = []

        for _ in range(0, self.args.num_trials):
            random.shuffle(self.words)
            red_words.append(self.words[:12])
            blue_words.append(self.words[12:24])
            black_words.append([self.words[24]])

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

        shuffled_embeddings = self.args.embeddings
        random.shuffle(shuffled_embeddings)

        for embedding_type in shuffled_embeddings:
            embedding_trial_number = 0
            debug_file_path = None
            if self.args.debug is True or self.args.debug_file != None:
                debug_file_path = (embedding_type + "-" + datetime.now().strftime("%m-%d-%Y-%H.%M.%S") + ".txt") if self.args.debug_file == None else self.args.debug_file
                # Create directory to put debug files if it doesn't exist
                if not os.path.exists('debug_output'):
                    os.makedirs('debug_output')
                debug_file_path = os.path.join('debug_output', debug_file_path)
                # print("Writing debug output to", debug_file_path)

            configuration = CodenamesConfiguration(
                verbose=self.args.verbose,
                visualize=self.args.visualize,
                split_multi_word=self.args.split_multi_word,
                disable_verb_split=self.args.disable_verb_split,
                debug_file=debug_file_path,
                length_exp_scaling=self.args.length_exp_scaling,
                use_heuristics=(not self.args.no_heuristics),
                single_word_label_scores=self.args.single_word_label_scores,
                use_kim_scoring_function=self.args.use_kim_scoring_function,
                babelnet_api_key=self.args.babelnet_api_key,
            )

            self.board = Board(red_words[0], blue_words[0], black_words[0])

# represents the environment that the guesser experiences.
# one step is the guesser taking in a hint
# and returning a board index. the cardinality is preset
class GuesserEnv(gym.Env):
    def __init__(self, words, n, args):
            num_words = len(words)
            self.words_per_clue = args.words_per_clue
            self.possible_agents = ["guesser"]
            self.action_space = env.spaces.Discrete(25)
            self.observation_space = spaces.Dict(
                {
                    "words": spaces.Box(0, num_words, shape=(25,), dtype=int),
                    "colors": spaces.Box(0, 2, shape=(25,), dtype=int),
                    "chosen": spaces.Box(0, 1, shape=(25,), dtype=int),
                }
            )
            self.board = None
            self.game = None
            self.words = words
            self.args = args
            self.reset()
    def step(self, action):
            # map it to words
            done = False
            reward = 0
            for i in action:
                word = self.words[action]
                result = self.board.choose(word)
                if result == 1 or result == -1:
                    done = True
                reward += result
            info = {}
            state = self.get_state()
            return state, reward, done, info
    #get the state from the board state
    def get_state(self):
        return self.board.get_state()
    # def get_clues(self):
    def display(self):
        self.board.print()
    def get_action(self):
        best_scores, best_clues, best_board_words_for_clue = self.game.get_clue(2, 1)
    def reset(self):
        red_words = []
        blue_words = []
        black_words = []

        for _ in range(0, self.args.num_trials):
            random.shuffle(self.words)
            red_words.append(self.words[:12])
            blue_words.append(self.words[12:24])
            black_words.append([self.words[24]])

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

        shuffled_embeddings = self.args.embeddings
        random.shuffle(shuffled_embeddings)

        for embedding_type in shuffled_embeddings:
            embedding_trial_number = 0
            debug_file_path = None
            if self.args.debug is True or self.args.debug_file != None:
                debug_file_path = (embedding_type + "-" + datetime.now().strftime("%m-%d-%Y-%H.%M.%S") + ".txt") if self.args.debug_file == None else self.args.debug_file
                # Create directory to put debug files if it doesn't exist
                if not os.path.exists('debug_output'):
                    os.makedirs('debug_output')
                debug_file_path = os.path.join('debug_output', debug_file_path)
                # print("Writing debug output to", debug_file_path)

            configuration = CodenamesConfiguration(
                verbose=self.args.verbose,
                visualize=self.args.visualize,
                split_multi_word=self.args.split_multi_word,
                disable_verb_split=self.args.disable_verb_split,
                debug_file=debug_file_path,
                length_exp_scaling=self.args.length_exp_scaling,
                use_heuristics=(not self.args.no_heuristics),
                single_word_label_scores=self.args.single_word_label_scores,
                use_kim_scoring_function=self.args.use_kim_scoring_function,
                babelnet_api_key=self.args.babelnet_api_key,
            )
            
            self.board = Board(red_words[0], blue_words[0], black_words[0])