import sys


sys.path.insert(0, "../")

stopwords = [
    'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'get', 'put',
]

idf_lower_bound = 0.0006
default_single_word_label_scores = (1, 1.1, 1.1, 1.2)

"""
Configuration for running the game
"""
class CodenamesConfiguration(object):
    def __init__(
        self,
        verbose=False,
        visualize=False,
        split_multi_word=True,
        disable_verb_split=True,
        debug_file=None,
        length_exp_scaling=None,
        use_heuristics=True,
        single_word_label_scores=default_single_word_label_scores,
        use_kim_scoring_function=False,
        babelnet_api_key=None,
    ):
        self.verbose = verbose
        self.visualize = visualize
        self.split_multi_word = split_multi_word
        self.disable_verb_split = disable_verb_split
        self.debug_file = debug_file
        self.length_exp_scaling = length_exp_scaling
        self.use_heuristics = use_heuristics
        self.single_word_label_scores = tuple(single_word_label_scores)
        self.use_kim_scoring_function = use_kim_scoring_function
        self.babelnet_api_key = babelnet_api_key

    def description(self):
        return (
            "<verbose: " + str(self.verbose) +
            ",visualize: " + str(self.visualize) +
            ",split multi-word clues: " + str(self.split_multi_word) +
            ",disable verb split: " + str(self.disable_verb_split) +
            ",length exp scaling: " + str(self.length_exp_scaling) +
            ",use heuristics: " + str(self.use_heuristics) +
            ",use kim scoring function: " + str(self.use_kim_scoring_function) +
            ">"
        )