import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import stanza
from itertools import chain

class Normalize():
    def __init__(self) -> None:
        stanza.download('ru')
        self.nlp = stanza.Pipeline(lang='ru', use_gpu=True)
        nltk.download('russian_stopwords')
        self.russian_stopwords = nltk.corpus.stopwords.words('russian')


    def normalize_data(self, content):
        new_data = []
        i = 0
        for item in content:
            i+=1
            print(f'\r{i}/{len(content)}', end='')
            data = self.nlp(item)
            tokens = []
            raw = self.nlp(item).sentences

            for sen in raw:
                for token in sen.words:
                    if token.upos in ('PUNCT', 'ADP') or token.text in self.russian_stopwords:
                        continue
                    tokens.append(token)

            new_data.append(self.normalize(self, tokens))
            if i == 1000: break
        return new_data
    

    def normalize(self, tokens):
            if len(tokens) == 0:
                return ""
            
            new_sentence = tokens[0].lemma
            if len(tokens) == 1:
                return new_sentence
            
            new_sentence += ' '
            for token in tokens[1:]:
                if token.upos in ('PUNCT', 'ADP') or token.text in self.russian_stopwords:
                    continue

                new_sentence += token.lemma + ' '
            
            return new_sentence
    

    
    