import random
from termcolor import colored
import numpy as np

class Board:
    def __init__(self, red_words, blue_words, black_words):
        # print("lens", len(red_words), len(blue_words), len(black_words))
        if len(red_words) != 12 or len(blue_words) != 12 or len(black_words) != 1:
            raise Exception("incorrect number of words in a category for board.")
        self.red_words = red_words
        self.blue_words = blue_words
        self.black_words = black_words
        self.words = red_words + blue_words + black_words
        random.shuffle(self.words)
        self.chosen = {}
        for word in self.words:
            self.chosen[word] = 0
        self.colors = []
        for word in self.words:
            self.colors.append(self.color_int(word))
    def color(self, idx):
        word = self.words[idx]
        if word in self.blue_words:
            return "blue"
        elif word in self.red_words:
            return "red"
        else:
            return "grey"
    def get_color(self, word):
        if word in self.blue_words:
            return "blue"
        elif word in self.red_words:
            return "red"
        else:
            return "black"
    def color_int(self, word):
        #blue
        if word in self.blue_words:
            return 0
        #red
        elif word in self.red_words:
            return 1
        #black
        else:
            return 2

    def access(self, idx):
        return self.words[idx]
    def print(self):
        for i in range(5):
            for j in range(5):
                word = self.access(i*5 + j)
                chosen = self.chosen(word)
                if chosen:
                    print(colored(self.access(i*5 + j), self.color(i*5 + j), attrs=["bold"]) + " ", end='')
                else:
                    print(colored(self.access(i*5 + j), self.color(i*5 + j)) + " ", end='')
            print()
    def choose(self, word):
        self.chosen[word] = 1
        if word == self.black_words[0]:
            return -1
        if word in self.red_words:
            return -0.5
        if self.is_finished() == 1:
            return 1
        else:
            return 0
    def chosen(self, word):
        return self.chosen[word]
    def get_state(self):
        return {
            "words": np.array(self.words),
            "colors": np.array(self.colors),
            "chosen": np.array(list(self.chosen.values()))
        }
    def is_finished(self):
        blueresult = [value for key, value in self.chosen.items() if key not in self.black_words and key in self.blue_words]
        redresult = [value for key, value in self.chosen.items() if key not in self.black_words and key in self.red_words]

        # if they're all 1 besides for black
        if len(list(set(blueresult))) == 1 and blueresult[0] == 1:
            return 1
        return 0