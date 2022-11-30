import random
from termcolor import colored

class Board:
    def __init__(self, red_words, blue_words, black_words):
        # print("lens", len(red_words), len(blue_words), len(black_words))
        if len(red_words) != 12 or len(blue_words) != 12 or len(black_words) != 1:
            raise Exception("incorrect number of words in a category for board.")
        self.red_words = red_words
        self.blue_words = blue_words
        self.black_words = black_words
        self.all_words = red_words + blue_words + black_words
        random.shuffle(self.all_words)
        # self.chosen = {self.all_words: 0}
    def color(self, idx):
        word = self.all_words[idx]
        if word in self.blue_words:
            return "blue"
        elif word in self.red_words:
            return "red"
        else:
            return "grey"
    
    def access(self, idx):
        return self.all_words[idx]
    def print(self):
        for i in range(5):
            for j in range(5):
                print(colored(self.access(i*5 + j), self.color(i*5 + j), attrs=["bold"]) + " ", end='')
            print()
    # def choose(self, idx):
    #     self.chosen[self.access(idx)] = 1