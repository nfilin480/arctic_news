import math
from collections import Counter, defaultdict
from typing import Any
import pandas as pd
#from tqdm import tqdm
#from textRank import TextRankSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from sklearn.cluster import KMeans
import numpy as np
import tqdm
from textRank import TextRankSummarizer

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append('E:\\IT_projects\\arctic_news\\preprocessor\\')
#sys.path.append('E:\\IT_projects\\arctic_news\\datasets')
from syntax_analyzer import SyntaxAnalyzer

class LexRankSummarizer(TextRankSummarizer):
    """
    LexRank.
    Оригинальная статья: https://arxiv.org/abs/1109.2128
    """

    def __init__(self, texts, threshold=None, *args, **kwagrs):
        super().__init__(*args, **kwagrs)

        self.idfs = self._collect_idfs(texts)
        self.set_sim_function(lambda x1, x2: self._lexrank_similarity_function(x1, x2))
        self.set_threshold(threshold)

    def _collect_idfs(self, texts):
        analyzer = SyntaxAnalyzer()
        documents_cnt = Counter()
        c = 0
        for text in texts:
            c +=1
            print(f'c = {c}')
            doc = analyzer(text)
            sentences = analyzer.get_sentences(doc, normalize=True, upos=['NOUN', 'VERB', 'AUX', 'ADJ'])
            for sentence in sentences:
                for token in set(sentence.split(' ')):
                    documents_cnt[token] += 1
        idfs = defaultdict(float)
        for token, cnt in documents_cnt.items():
            idfs[token] = math.log(len(texts) / cnt)
        return idfs
      
    def _lexrank_similarity_function(self, tokens1, tokens2):
        tf1_all = Counter(tokens1)
        tf2_all = Counter(tokens2)
        tf = 0
        for token, tf1 in tf1_all.items():
            tf2 = tf2_all.get(token, 0)
            idf = self.idfs[token]
            tf += tf1 * tf2 * idf * idf
        norm1 = sum(((tf1 * self.idfs[token]) ** 2 for token, tf1 in tf1_all.items()))
        norm2 = sum(((tf2 * self.idfs[token]) ** 2 for token, tf2 in tf2_all.items()))
        if norm1 < 1e-7 or norm2 < 1e-7:
            return 0.0
        return tf / (math.sqrt(norm1) * math.sqrt(norm2))

    def __call__(self, text, target_sentences_count):
        return super().__call__(text, target_sentences_count)



        # Применение кластеризации KMeans к матрице TF-IDF
        

        # Получение списка ключевых слов и их значения TF-IDF для первого документа
        #feature_names = tfidf_vectorizer.get_feature_names_out()
        #tfidf_scores = tfidf_matrix.toarray()[0]
        #print(tfidf_scores, feature_names)

        # Сортировка слов по значениям TF-IDF
        #scores = zip(tfidf_scores, feature_names)
        #sorted_keywords = [word for _, word in sorted(zip(tfidf_scores, feature_names), reverse=True)]

        #print("Ключевые слова:", sorted_keywords)

        #return sorted_keywords[:total_sentences]
        #return tfidf_matrix#, tfidf_scores





texts = pd.read_csv('E:\\IT_projects\\arctic_news\\datasets\\xlsum500.csv')

# Вот так сделано для ускорения, для лучшего качества нужно подавать все тексты
#texts = texts

lex_rank = LexRankSummarizer(texts['text'][:10], verbose=True, threshold=None)
summary = lex_rank(texts['text'][0], 3)
print()
print("Итоговый реферат: {}".format(summary))
print(f"Реферат из датасета: {texts['summary'][0]}")

#lex_rank = LexRankSummarizer()
#result = lex_rank(texts['text'][0], 3)
#for a, b in result:
#    print(a, b)
#print(texts['summary'][0])