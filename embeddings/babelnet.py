import gzip
import os
import re
import requests
import string
import numpy
import pickle

# Gensim
from gensim.utils import simple_preprocess
from gensim.models import KeyedVectors

# Graphing
import networkx as nx
from networkx.exception import NodeNotFound
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt

babelnet_relationships_limits = {
	"HYPERNYM": float("inf"),
	"OTHER": 0,
	"MERONYM": 20,
	"HYPONYM": 20,
}

punctuation = re.compile("[" + re.escape(string.punctuation) + "]")


class Babelnet(object):

	def __init__(self, configuration=None):
		# constants
		self.VERB_SUFFIX = "v"
		self.NOUN_SUFFIX = "n"
		self.ADJ_SUFFIX = "a"

		#  File paths to cached babelnet query results
		self.file_dir = 'babelnet_v6/'
		self.synset_main_sense_file = self.file_dir + 'synset_to_main_sense.txt'
		self.synset_senses_file = self.file_dir + 'synset_to_senses.txt'
		self.synset_glosses_file = self.file_dir + 'synset_to_glosses.txt'
		self.synset_domains_file = self.file_dir + 'synset_to_domains.txt'
		self.synset_metadata_file = self.file_dir + 'synset_to_metadata.txt'

		# Initialize variables
		self.configuration = configuration
		# {codeword : {nearest_neighbor : babelnet_synset_id }}
		self.nn_to_synset_id = dict()
		# {codeword : {nearest_neighbor : babelnet_domain_label }}
		self.nn_to_domain_label = dict()
		self.graphs = dict()
		self.dictionary_definitions = dict()
		(
			self.synset_to_main_sense,
			self.synset_to_senses,
			self.synset_to_definitions,
			self.synset_to_domains,
			self.synset_to_metadata,
		) = self._load_synset_data_v5()

		self.word2vec_model = self._get_word2vec()

		self.dict2vec_embeddings_file = 'data/word_to_dict2vec_embeddings'
		self.word_to_dict2vec_embeddings = self._get_dict2vec()

	"""
	Pre-process steps
	"""

	def _load_synset_data_v5(self):
		"""Load synset_to_main_sense"""
		synset_to_main_sense = {}
		synset_to_senses = {}
		synset_to_definitions = {}
		synset_to_domains = {}
		synset_to_metadata = {}
		if os.path.exists(self.synset_main_sense_file):
			with open(self.synset_main_sense_file, "r") as f:
				for line in f:
					parts = line.strip().split("\t")
					synset, main_sense = parts[0], parts[1]
					synset_to_main_sense[synset] = main_sense
					synset_to_senses[synset] = set()
					# synset_to_domains[synset] = dict()
					# TODO: uncomment and remove initialization below
		if os.path.exists(self.synset_senses_file):
			with open(self.synset_senses_file, "r") as f:
				for line in f:
					parts = line.strip().split("\t")
					assert len(parts) == 5
					synset, full_lemma, simple_lemma, source, pos = parts
					if source == "WIKIRED":
						continue
					synset_to_senses[synset].add(simple_lemma)
		if os.path.exists(self.synset_glosses_file):
			with open(self.synset_glosses_file, "r") as f:
				for line in f:
					parts = line.strip().split("\t")
					assert len(parts) == 3
					synset, source, definition = parts
					if synset not in synset_to_definitions:
						synset_to_definitions[synset] = set()
					if source == "WIKIRED":
						continue
					synset_to_definitions[synset].add(definition)
		if os.path.exists(self.synset_domains_file):
			with open(self.synset_domains_file, "r") as f:
				for line in f:
					parts = line.strip().split("\t")
					assert len(parts) == 3
					synset, domain, score = parts
					if synset not in synset_to_domains:
						synset_to_domains[synset] = dict()
					if domain == "NONE":
						continue
					score == float(score)
					if domain in synset_to_domains[synset]:
						synset_to_domains[synset][domain] = max(
							synset_to_domains[synset][domain], score
						)
					else:
						synset_to_domains[synset][domain] = score
		if os.path.exists(self.synset_metadata_file):
			with open(self.synset_metadata_file, "r") as f:
				for line in f:
					parts = line.strip().split("\t")
					assert len(parts) == 3
					synset, keyConcept, synsetType = parts
					synset_to_metadata[synset] = {
						"keyConcept": keyConcept,
						"synsetType": synsetType,
					}

		return (
			synset_to_main_sense,
			synset_to_senses,
			synset_to_definitions,
			synset_to_domains,
			synset_to_metadata,
		)


	def _get_word2vec(self):
		word2vec_model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
		return word2vec_model


	def _get_dict2vec(self):
		input_file = open(self.dict2vec_embeddings_file,'rb')
		word_to_dict2vec_embeddings = pickle.load(input_file)
		return word_to_dict2vec_embeddings

	"""
	Required codenames methods
	"""

	def get_weighted_nn(self, word, filter_entities=True):
		"""
		:param word: the codeword to get weighted nearest neighbors for
		returns: a dictionary mapping nearest neighbors (str) to distances from codeword (int)
		"""
		def should_add_relationship(relationship, level):
			if relationship != 'HYPERNYM' and level > 1:
				return False
			return relationship in babelnet_relationships_limits.keys() and \
				count_by_relation_group[relationship] < babelnet_relationships_limits[relationship]

		def _single_source_paths_filter(G, firstlevel, paths, cutoff, join):
			level = 0                  # the current level
			nextlevel = firstlevel
			while nextlevel and cutoff > level:
				thislevel = nextlevel
				nextlevel = {}
				for v in thislevel:
					for w in G.adj[v]:
						if len(paths[v]) >= 3 and G.edges[paths[v][1], paths[v][2]]['relationship'] != G.edges[v, w]['relationship']:
							continue
						if w not in paths:
							paths[w] = join(paths[v], [w])
							nextlevel[w] = 1
				level += 1
			return paths

		def single_source_paths_filter(G, source, cutoff=None):
			if source not in G:
				raise nx.NodeNotFound("Source {} not in G".format(source))

			def join(p1, p2):
				return p1 + p2
			if cutoff is None:
				cutoff = float('inf')
			nextlevel = {source: 1}     # list of nodes to check at next level
			# paths dictionary  (paths to key from source)
			paths = {source: [source]}
			return dict(_single_source_paths_filter(G, nextlevel, paths, cutoff, join))

		count_by_relation_group = {
			key: 0 for key in babelnet_relationships_limits.keys()}

		G = nx.DiGraph()
		with gzip.open(self.file_dir + word + '.gz', 'r') as f:
			for line in f:
				source, target, language, short_name, relation_group, is_automatic, level = line.decode(
					"utf-8").strip().split('\t')

				if should_add_relationship(relation_group, int(level)) and is_automatic == 'False':
					G.add_edge(source, target, relationship=short_name)
					count_by_relation_group[relation_group] += 1

		nn_w_dists = {}
		nn_w_synsets = {}
		nn_w_domains = {}
		dictionary_definitions_for_word = []
		with open(self.file_dir + word + '_synsets', 'r') as f:
			for line in f:
				synset = line.strip()
				try:
					# get all paths starting from source, filtered
					paths = single_source_paths_filter(
						G, source=synset, cutoff=10
					)
					# TODO: if we want to filter intermediate nodes, we need to call
					# get_cached_labels_from_synset_v5 for all nodes in path.
					if self.configuration.length_exp_scaling is not None:
						scaling_func = lambda x : self.configuration.length_exp_scaling ** x
					else:
						scaling_func = lambda x : x
					lengths = {neighbor: scaling_func(len(path))
							   for neighbor, path in paths.items()}
					# lengths = nx.single_source_shortest_path_length(
					#     G, source=synset, cutoff=10
					# )
				except NodeNotFound as e:
					print(e)
					continue
				for neighbor, length in lengths.items():
					neighbor_main_sense, neighbor_senses, _, neighbor_metadata = self.get_cached_labels_from_synset_v5(
						neighbor, get_domains=False, get_metadata=filter_entities)
					# Note: this filters entity clues, not intermediate entity nodes
					if filter_entities and neighbor_metadata["synsetType"] != "CONCEPT":
						if self.configuration.verbose:
							print("skipping non-concept:", neighbor, neighbor_metadata["synsetType"])
						continue

					split_multi_word = self.configuration.split_multi_word
					if self.configuration.disable_verb_split and synset.endswith(self.VERB_SUFFIX):
						split_multi_word = False

					single_word_labels = self.get_single_word_labels_v5(
						neighbor_main_sense,
						neighbor_senses,
						split_multi_word=split_multi_word,
					)
					for single_word_label, label_score in single_word_labels:
						if single_word_label not in nn_w_dists:
							nn_w_dists[single_word_label] = length * label_score
							nn_w_synsets[single_word_label] = neighbor
						else:
							if nn_w_dists[single_word_label] > (length * label_score):
								nn_w_dists[single_word_label] = length * label_score
								nn_w_synsets[single_word_label] = neighbor
				# get domains
				main_sense, sense, domains, _ = self.get_cached_labels_from_synset_v5(
					synset, get_domains=True)
				for domain, score in domains.items():
					nn_w_domains[domain] = float(score)

				# get definitions
				# TODO: some definitions are missing - we could call
				# get_cached_labels_from_synset_v5 to query for missing
				# definitions
				if synset in self.synset_to_definitions:
					dictionary_definitions_for_word.extend(
						word.lower().translate(str.maketrans('', '', string.punctuation))
						for definition in self.synset_to_definitions[synset]
						for word in definition.split()
					)

		self.nn_to_synset_id[word] = nn_w_synsets
		self.nn_to_domain_label[word] = nn_w_domains
		self.graphs[word] = G
		self.dictionary_definitions[word] = dictionary_definitions_for_word

		return {k: 1.0 / (v + 1) for k, v in nn_w_dists.items() if k != word}

	def rescale_score(self, chosen_words, clue, red_words):
		"""
		:param chosen_words: potential board words we could apply this clue to
		:param clue: potential clue
		:param red_words: opponent's words
		returns: using IDF and dictionary definition heuristics, how much to add to the score for this potential clue give these board words
		"""

		# factor in dictionary definition heuristic
		dict_definition_score = self._get_dictionary_definition_score(
			chosen_words, clue, red_words)

		word2vec_score = self._get_word2vec_score(chosen_words, clue, red_words)

		dict2vec_score = self._get_dict2vec_score(chosen_words, clue, red_words)

		if self.configuration.debug_file:
			with open(self.configuration.debug_file, 'a') as f:
				f.write(" ".join([str(x) for x in [
					" dict2vec score", round(dict2vec_score,3), "dictionary def score:", round(dict_definition_score,3), "word2vec score:", round(2*word2vec_score,3), "\n"
				]]))

		return (dict_definition_score) + (2*word2vec_score) + (dict2vec_score)

	"""
	Helper methods
	"""
	def _get_dict2vec_score(self, chosen_words, potential_clue, red_words):
		dict2vec_distances = []
		red_dict2vec_distances = []

		if potential_clue not in self.word_to_dict2vec_embeddings:
			if self.configuration.verbose:
				print("Potential clue word ", potential_clue, "not in dict2vec model")
			return 0.0

		potential_clue_embedding = self.word_to_dict2vec_embeddings[potential_clue]
		# TODO: change this to cosine distance
		for chosen_word in chosen_words:
			if chosen_word in self.word_to_dict2vec_embeddings:
				chosen_word_embedding = self.word_to_dict2vec_embeddings[chosen_word]
				euclidean_distance = numpy.linalg.norm(chosen_word_embedding-potential_clue_embedding)
				dict2vec_distances.append(euclidean_distance)

		for red_word in red_words:
			if red_word in self.word_to_dict2vec_embeddings:
				red_word_embedding = self.word_to_dict2vec_embeddings[red_word]
				red_euclidean_distance = numpy.linalg.norm(red_word_embedding-potential_clue_embedding)
				red_dict2vec_distances.append(red_euclidean_distance)
			else:
				print(f"red word {red_word} not in dict2vec")
		#TODO: is average the best way to do this
		return 1/(sum(dict2vec_distances)/len(dict2vec_distances)) - 1/(min(red_dict2vec_distances))

	def _get_word2vec_score(self, chosen_words, potential_clue, red_words):

		word2vec_similarities = []
		red_word2vec_similarities = []
		if potential_clue not in self.word2vec_model:
			if self.configuration.verbose:
				print("Potential clue word ", potential_clue, "not in Google news word2vec model")
			return 0.0

		# TODO: cache this info in pre-training
		for chosen_word in chosen_words:
			if chosen_word in self.word2vec_model:
				word2vec_similarities.append(self.word2vec_model.similarity(chosen_word, potential_clue))
		for red_word in red_words:
			if red_word in self.word2vec_model:
				red_word2vec_similarities.append(self.word2vec_model.similarity(red_word, potential_clue))
			else:
				print(f"red word {red_word} not in word2vec")
		#TODO: is average the best way to do this
		return sum(word2vec_similarities)/len(word2vec_similarities) - max(red_word2vec_similarities)

	def _get_dictionary_definition_score(self, chosen_words, potential_clue, red_words):
		# the dictionary definitions of words (as given from their babelnet synset)
		# used as a heuristic for candidate clue words

		is_in_chosen_words_dict_definition = 0.0
		is_in_red_words_dict_definition = 0.0
		for word in chosen_words:
			if potential_clue in self.dictionary_definitions[word]:
				is_in_chosen_words_dict_definition += 1.0

		for word in red_words:
			if potential_clue in self.dictionary_definitions[word]:
				is_in_red_words_dict_definition += 1.0

		return is_in_chosen_words_dict_definition - is_in_red_words_dict_definition

	"""
	Babelnet methods
	"""

	def get_cached_labels_from_synset_v5(
			self, synset, get_domains=False, get_metadata=False
	):
		"""This actually gets the main_sense but also writes all senses/glosses"""
		if (
				synset not in self.synset_to_main_sense
				or (get_domains and synset not in self.synset_to_domains)
				or (get_metadata and synset not in self.synset_to_metadata)
		):
			print("getting query", synset)
			# assert False
			labels_json = self.get_labels_from_synset_v5_json(synset)
			self.write_synset_labels_v5(synset, labels_json)

			# filtered_labels = [label for label in labels if len(label.split("_")) == 1 or label.split("_")[1][0] == '(']
			# self.synset_to_labels[synset] = self.get_random_n_labels(filtered_labels, 3) or synset
		# sliced_labels = self.synset_to_labels[synset]
		main_sense = self.synset_to_main_sense[synset]
		senses = self.synset_to_senses[synset]
		domains = self.synset_to_domains[synset] if get_domains else {}
		metadata = self.synset_to_metadata[synset] if get_metadata else {}
		return main_sense, senses, domains, metadata

	def write_synset_labels_v5(self, synset, json):
		"""Write to synset_main_sense_file, synset_senses_file, and synset_glosses_file"""
		if synset not in self.synset_to_main_sense:
			with open(self.synset_main_sense_file, "a") as f:
				if "mainSense" not in json:
					if self.configuration.verbose:
						print("no main sense for", synset)
					main_sense = synset
				else:
					main_sense = json["mainSense"]
				f.write("\t".join([synset, main_sense]) + "\n")
				self.synset_to_main_sense[synset] = main_sense

		if synset not in self.synset_to_senses:
			self.synset_to_senses[synset] = set()
			with open(self.synset_senses_file, "a") as f:
				self.synset_to_senses[synset] = set()
				if "senses" in json:
					for sense in json["senses"]:
						properties = sense["properties"]
						line = [
							synset,
							properties["fullLemma"],
							properties["simpleLemma"],
							properties["source"],
							properties["pos"],
						]
						f.write("\t".join(line) + "\n")
						if properties["source"] != "WIKIRED":
							self.synset_to_senses[synset].add(properties["simpleLemma"])

		if synset not in self.synset_to_definitions:
			self.synset_to_definitions[synset] = set()
			with open(self.synset_glosses_file, "a") as f:
				if "glosses" in json:
					if len(json["glosses"]) == 0:
						f.write("\t".join([synset, "NONE", "NONE"]) + "\n")
					else:
						for gloss in json["glosses"]:
							line = [synset, gloss["source"], gloss["gloss"]]
							f.write("\t".join(line) + "\n")
							if gloss["source"] != "WIKIRED":
								self.synset_to_definitions[synset].add(gloss["gloss"])

		if synset not in self.synset_to_domains:
			self.synset_to_domains[synset] = dict()
			with open(self.synset_domains_file, "a") as f:
				if "domains" in json:
					for domain, score in json["domains"].items():
						if domain in self.synset_to_domains[synset]:
							self.synset_to_domains[synset][domain] = max(
								self.synset_to_domains[synset][domain], score
							)
						else:
							self.synset_to_domains[synset][domain] = score
						f.write("\t".join([synset, domain, str(score)]) + "\n")
				if "domains" not in json or len(json["domains"]) == 0:
					f.write("\t".join([synset, "NONE", "-100"]) + "\n")

		if synset not in self.synset_to_metadata:
			with open(self.synset_metadata_file, "a") as f:
				keyConcept = "NONE"
				synsetType = "NONE"
				if "bkeyConcepts" in json:
					keyConcept = str(json["bkeyConcepts"])
				if "synsetType" in json:
					synsetType = json["synsetType"]
				f.write("\t".join([synset, keyConcept, synsetType]) + "\n")
				self.synset_to_metadata[synset] = {
					"keyConcept": keyConcept,
					"synsetType": synsetType,
				}



	def get_labels_from_synset_v5_json(self, synset):
		url = 'https://babelnet.io/v5/getSynset'
		params = {
			'id': synset,
			'key': 'e3b6a00a-c035-4430-8d71-661cdf3d5837'
		}
		headers = {'Accept-Encoding': 'gzip'}
		res = requests.get(url=url, params=params, headers=headers)
		return res.json()

	def parse_lemma_v5(self, lemma):
		lemma_parsed = lemma.split('#')[0]
		parts = lemma_parsed.split('_')
		single_word = len(parts) == 1 or parts[1].startswith('(')
		return parts[0], single_word

	def get_single_word_labels_v5(self, lemma, senses, split_multi_word=False):
		""""""
		main_single, main_multi, other_single, other_multi = self.configuration.single_word_label_scores
		single_word_labels = []
		parsed_lemma, single_word = self.parse_lemma_v5(lemma)
		if single_word:
			single_word_labels.append((parsed_lemma, main_single))
		elif split_multi_word:
			single_word_labels.extend(
				zip(parsed_lemma.split("_"), [main_multi for _ in parsed_lemma.split("_")])
			)

		for sense in senses:
			parsed_lemma, single_word = self.parse_lemma_v5(sense)
			if single_word:
				single_word_labels.append((parsed_lemma, other_single))
			elif split_multi_word:
				single_word_labels.extend(
					zip(parsed_lemma.split("_"), [other_multi for _ in parsed_lemma.split("_")])
				)
		if len(single_word_labels) == 0:
			# can only happen if split_multi_word = False
			assert not split_multi_word
			return [(lemma.split("#")[0], 1)]
		return single_word_labels

	"""
	Visualization
	"""

	def get_intersecting_graphs(self, word_set, clue, split_multi_word):
		# clue is the word that intersects for this word_set
		# for example if we have the word_set (beijing, phoenix) and clue city,
		# this method will produce a directed graph that shows the path of intersection
		clue_graph = nx.DiGraph()
		word_set_graphs = [nx.DiGraph() for _ in word_set]

		if all([clue in self.nn_to_domain_label[word] for word in word_set]):
			# this clue came frome 2 domains, don't plot
			return

		for word in word_set:
			# Create graph that shows the intersection from the clue to the word_set
			if clue in self.nn_to_synset_id[word]:
				clue_synset = self.nn_to_synset_id[word][clue]  # synset of the clue
			else:
				if self.configuration.verbose:
					print("Clue ", clue, "not found for word", word)
				continue
			word_graph = self.graphs[word]
			shortest_path = []
			shortest_path_length = float("inf")
			with open(self.file_dir + word + '_synsets', 'r') as f:
				for line in f:
					synset = line.strip()
					try:
						path = nx.shortest_path(
							word_graph, synset, clue_synset)
						if len(path) < shortest_path_length:
							shortest_path_length = len(path)
							shortest_path = path
							shortest_path_synset = synset
					except:
						if self.configuration.verbose:
							print("No path between", synset, clue, clue_synset)

				# path goes from source (word in word_set) to target (clue)
				shortest_path_labels = []
				for synset in shortest_path:
					# TODO: add graph for domains?
					main_sense, senses, _, _ = self.get_cached_labels_from_synset_v5(
						synset, get_domains=False)
					if self.configuration.disable_verb_split and synset.endswith(self.VERB_SUFFIX):
						split_multi_word = False

					single_word_label = self.get_single_word_labels_v5(
						main_sense, senses, split_multi_word)[0][0]
					shortest_path_labels.append(single_word_label)

				if self.configuration.debug_file:
					with open(self.configuration.debug_file, 'a') as f:
						f.write(
							" ".join([str(x) for x in ["shortest path from", word, shortest_path_synset,
							"to clue", clue, clue_synset, ":", shortest_path_labels, "\n"]])
						)
				else:
					print("shortest path from", word, shortest_path_synset,
						"to clue", clue, clue_synset, ":", shortest_path_labels)

				formatted_labels = [label.replace(
					' ', '\n') for label in shortest_path_labels]
				formatted_labels.reverse()
				nx.add_path(clue_graph, formatted_labels)

		self.draw_graph(clue_graph, ('_').join(
			[clue] + [word for word in word_set]))

	def draw_graph(self, graph, graph_name, get_labels=False):
		write_dot(graph, 'test.dot')
		pos = graphviz_layout(graph, prog='dot')

		# if we need to get labels for our graph (because the node text is synset ids)
		# we will create a dictionary { node : label } to pass into our graphing options
		nodes_to_labels = dict()
		current_labels = nx.draw_networkx_labels(graph, pos=pos)

		if get_labels:
			for synset_id in current_labels:
				main_sense, senses, _, _ = self.get_cached_labels_from_synset_v5(
					synset_id, get_domains=True)
				nodes_to_labels[synset_id] = main_sense
		else:
			nodes_to_labels = {label: label for label in current_labels}

		plt.figure()
		options = {
			'node_color': 'white',
			'line_color': 'black',
			'linewidths': 0,
			'width': 0.5,
			'pos': pos,
			'with_labels': True,
			'font_color': 'black',
			'font_size': 3,
			'labels': nodes_to_labels,
		}
		nx.draw(graph, **options)

		if not os.path.exists('intersection_graphs'):
			os.makedirs('intersection_graphs')
		filename = 'intersection_graphs/' + graph_name + '.png'
		# margins
		plot_margin = 0.35
		x0, x1, y0, y1 = plt.axis()
		plt.axis((x0 - plot_margin,
				  x1 + plot_margin,
				  y0 - plot_margin,
				  y1 + plot_margin))
		plt.savefig(filename, dpi=300)
		plt.close()
