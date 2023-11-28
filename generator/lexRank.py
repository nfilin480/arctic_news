import math
from collections import Counter, defaultdict
from typing import Any
import pandas as pd
#from tqdm import tqdm
#from textRank import TextRankSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append('C:\\Users\\sante\\OneDrive\\Документы\\labs\\arctic_news\\arctic_news\\preprocessor')
from syntax_analyzer import SyntaxAnalyzer

class LexRankSummarizer():
    """
    LexRank.
    Оригинальная статья: https://arxiv.org/abs/1109.2128
    """

    def __init__(self):
        self.analyzer = SyntaxAnalyzer()


    def __call__(self, text: str, total_sentences: int) -> Any:
        doc = self.analyzer(text)
        sentences = self.analyzer.get_sentences(doc, normalize=False, upos=[])

        # Создание объекта TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Применение TF-IDF к текстовым данным
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

        # Получение списка ключевых слов и их значения TF-IDF для первого документа
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Сортировка слов по значениям TF-IDF
        sorted_keywords = [word for _, word in sorted(zip(tfidf_scores, feature_names), reverse=True)]

        print("Ключевые слова:", sorted_keywords)

        return sorted_keywords[:total_sentences]





texts = pd.read_csv('../')

lex_rank = LexRankSummarizer()
print(lex_rank(texts['text'][0], 3))