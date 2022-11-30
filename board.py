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
        return self.all_words[idx]
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
    def chosen(self, word):
        return self.chosen[word]

    def get_state(self):
        return {
            "words": self.words,
            "colors": self.colors,
            "chosen": self.chosen
        }