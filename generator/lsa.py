import math
from numpy.linalg import svd
import numpy as np
import pandas as pd
import sys

sys.path.append('E:\\IT_projects\\arctic_news\\preprocessor')
from syntax_analyzer import SyntaxAnalyzer


class LsaSummarizer:
    """
    Латентно-семантический анализ для реферирования.
    Основано на: https://github.com/miso-belica/sumy/blob/main/sumy/summarizers/lsa.py
    Оригинальная статья: https://www.cs.bham.ac.uk/~pxt/IDA/text_summary.pdf
    """
    def __init__(
        self,
        verbose,
        preprocessing_function = SyntaxAnalyzer()
    ):
        self.verbose = verbose
        self.preprocessing_function = preprocessing_function

    def __call__(self, text, target_sentences_count):

        doc = self.preprocessing_function(text)
        original_sentences = self.preprocessing_function.get_sentences(doc, normalize=False, upos=[])
        #print(len(original_sentences))
        tokenized_sentences = self.preprocessing_function.get_sentences(doc, normalize=True, upos=['NOUN', 'VERB', 'AUX', 'ADJ'])

        #original_sentences = [s for s in sentenize(text)]
        #tokenized_sentences = [tokenize_sentence(s) for s in original_sentences]

        # Словарь для последующего построения матрицы
        vocabulary = {token for sentence in tokenized_sentences for token in sentence}
        vocabulary = {word: idx for idx, word in enumerate(vocabulary)}
        if not vocabulary:
            return ""

        # Собственно построение матрицы
        matrix = self._create_matrix(tokenized_sentences, vocabulary)
        matrix = self._norm_matrix(matrix)

        # Сингулярное разложение
        _, sigma, v_matrix = svd(matrix, full_matrices=False)
      
        # Оставляем только важные тематики
        min_dimensions = max(3, target_sentences_count)
        topics_weights = [s**2 for s in sigma[:min_dimensions]]
        print("Веса важных тематик:", topics_weights)

        # Смотрим, как предложения в представлены в этих важных тематиках
        ranks = []
        for sentence_column, s in zip(v_matrix.T, original_sentences):
            sentence_column = sentence_column[:min_dimensions]
            print("ПРЕДЛОЖЕНИЕ: веса тематик: {}, текст: {}".format(sentence_column, s))
            rank = sum(s*v**2 for s, v in zip(topics_weights, sentence_column))
            ranks.append(math.sqrt(rank))

        indices = list(range(len(tokenized_sentences)))
        indices = [idx for _, idx in sorted(zip(ranks, indices), reverse=True)]
        indices = indices[:target_sentences_count]
        indices.sort()
        return " ".join([original_sentences[idx] for idx in indices])

    def _create_matrix(self, sentences, vocabulary):
        """
        Создание матрицы инцидентности
        """
        words_count = len(vocabulary)
        sentences_count = len(sentences)

        matrix = np.zeros((words_count, sentences_count))
        for col, sentence in enumerate(sentences):
            for word in sentence:
                row = vocabulary[word]
                matrix[row, col] += 1
        return matrix

    def _norm_matrix(self, matrix):
        """
        Нормировка матрицы инцидентности
        """
        max_word_frequencies = np.max(matrix, axis=0)
        rows, cols = matrix.shape
        for row in range(rows):
            for col in range(cols):
                max_word_frequency = max_word_frequencies[col]
                if max_word_frequency != 0:
                    matrix[row, col] /= max_word_frequency
        return matrix
    


a = LsaSummarizer(verbose=True)

data = pd.read_csv('E:\\IT_projects\\arctic_news\\datasets\\xlsum500.csv')

print(a(data['text'][0], 3))
print(f"Summary dataset: {data['summary'][0]}")