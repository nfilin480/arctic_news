import math
import numpy as np
import networkx as nx
import seaborn as sns
from scipy.linalg import eig
from matplotlib import pyplot as plt
import razdel
import pandas as pd
#from preprocessor.normalize import Normalize
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append('E:\\IT_projects\\arctic_news\\preprocessor')
from syntax_analyzer import SyntaxAnalyzer


#def text_rank_preprocessing(sentence):
#    return SyntaxAnalyzer.get_sentences(sentence)


def text_rank_similarity(tokens1, tokens2):
    intersection_size = sum(tokens2.count(w) for w in tokens1)
    if intersection_size == 0:
        return 0.0

    if len(tokens1) <= 1 and len(tokens2) <= 1:
        return intersection_size

    assert len(tokens1) > 0 and len(tokens2) > 0
    norm = math.log(len(tokens1)) + math.log(len(tokens2))
    return intersection_size / norm


class TextRankSummarizer:
    """
    TextRank.
    Основано на: https://github.com/miso-belica/sumy/blob/main/sumy/summarizers/text_rank.py
    Оригинальная статья: https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
    """

    def __init__(
        self,
        damping = 0.85,
        epsilon = 1e-4,
        niter = 100,
        preprocessing_function = SyntaxAnalyzer(),
        similarity_function = text_rank_similarity,
        verbose=False
    ):
        self.damping = damping
        self.epsilon = epsilon
        self.niter = niter
        self.preprocessing_function = SyntaxAnalyzer()
        self.similarity_function = similarity_function
        self.threshold = None
        self.verbose = True

    def __call__(self, text, target_sentences_count):
        doc = self.preprocessing_function(text)
        original_sentences = self.preprocessing_function.get_sentences(doc, normalize=False, upos=[])
        print(len(original_sentences))
        sentences = self.preprocessing_function.get_sentences(doc, normalize=True, upos=['NOUN', 'VERB', 'AUX', 'ADJ'])

        graph = self._create_graph(sentences)  
        graph = self._apply_threshold(graph)
        if self.verbose:
            plt.figure(figsize=(15,10))
            sns.heatmap(graph, annot=True, fmt=".2f").set_title("Матрица схожести предложений")
        norm_graph = self._norm_graph(graph)
        ranks = self._iterate(norm_graph)

        if self.verbose:
            print("Значимости: {}".format(ranks))

            # Можно считать PageRank библиотечными методами.
            # При запуске на оригинальном графе должно быть то же самое.
            nx_graph = nx.from_numpy_array(graph)
            indices = list(range(len(sentences)))
            nx_ranks = nx.pagerank(nx_graph)
            nx_ranks = [ranks[i] for i in indices]
            assert np.all(np.isclose(nx_ranks, ranks))
            print("Проверка через NetworkX в порядке!")
        
        indices = list(range(len(sentences)))
        indices = [idx for _, idx in sorted(zip(ranks, indices), reverse=True)]
        indices = indices[:target_sentences_count]
        indices.sort()
        return " ".join([original_sentences[idx] for idx in indices])

    def set_sim_function(self, func):
        self.similarity_function = func

    def set_preprocessing_function(self, func):
        self.preprocessing_function = func
    
    def set_threshold(self, threshold):
        self.threshold = threshold

    def _create_graph(self, sentences):
        """ Сборка изначального графа схожостей """
        sentences_count = len(sentences)
        graph = np.zeros((sentences_count, sentences_count))
        for sentence_num1, sentence1 in enumerate(sentences):
            for sentence_num2 in range(sentence_num1, sentences_count):
                sentence2 = sentences[sentence_num2]
                sim = self.similarity_function(sentence1, sentence2)
                graph[sentence_num1, sentence_num2] = sim
                graph[sentence_num2, sentence_num1] = sim
        return graph
    
    def _apply_threshold(self, graph):
        """ Обрезка графа по порогу, понадобится в LexRank """
        if self.threshold is None:
            return graph
        graph[graph < self.threshold] = 0.0
        return graph

    def _norm_graph(self, graph):
        """
        Нормировка по строкам, потому что ниже p_vector - вектор, а не столбец.
        Если бы p_vector был столбцом, надо было бы нормировать по столбцам.
        """
        norm = graph.sum(axis=1)[:, np.newaxis]
        norm_graph = graph / (norm + 1e-7)
        #assert np.isclose(np.sum(norm_graph[0, :]), 1.0)
        #assert np.all(np.isclose(norm_graph.sum(axis=1), np.ones((graph.shape[0], ))))
        return norm_graph

    def _iterate(self, matrix):
        sentences_count = len(matrix)
        iter = 0
        lambda_val = 0.1
        p_vector = np.full((sentences_count, ), 1.0 / sentences_count)
        random_transitions = np.full((sentences_count, ), 1.0 / sentences_count)

        transposed_matrix = matrix.T
        while iter < self.niter and lambda_val > self.epsilon:
            next_p = (1.0 - self.damping) * random_transitions + self.damping * np.dot(transposed_matrix, p_vector)
            lambda_val = np.linalg.norm(np.subtract(next_p, p_vector))
            p_vector = next_p
            iter += 1
        return p_vector

    def _power_method(self, matrix):
        sentences_count = len(matrix)
        iter = 0
        lambda_val = 0.1
        p_vector = np.full((sentences_count, ), 1.0 / sentences_count)
        
        transposed_matrix = matrix.T
        while iter < self.niter and lambda_val > self.epsilon:
            next_p = np.dot(transposed_matrix, p_vector)
            lambda_val = np.linalg.norm(np.subtract(next_p, p_vector))
            p_vector = next_p
            iter += 1
        return p_vector