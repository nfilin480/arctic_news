from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
from syntax_analyzer import SyntaxAnalyzer

class PrepareData():

    def __init__(self):
        model_name = 'Den4ikAI/sbert_large_mt_ru_retriever'
        self.model = SentenceTransformer(model_name)


    def __embed__(self, sentences: list[str]) -> torch.Tensor:
        return self.model.encode(sentences, show_progress_bar=False, convert_to_tensor=True)


    def __cos_sim__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)

        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def extract_summary(self, sentences: list[str], target_n = 3) -> list[str]:

        #len_original_summary = len(self.embeddings)

        input_embedding = self.__embed__(sentences)
        cos_matrix = self.__cos_sim__(input_embedding, self.embeddings)

        indexes = torch.topk(cos_matrix.max(dim=1).values, target_n).indices

        result = [int(i in indexes) for i in range(len(sentences))]

        return result

    def __call__(self, news: list[str], summary: list[str], target_n = 3) -> list[str]:

        if len(news) < 3:
            return []

        self.embeddings = self.__embed__(summary)
        return self.extract_summary(news, target_n)
  

xlsum = load_dataset('csebuetnlp/xlsum', name='russian')['train']


processor = PrepareData()
analyzer = SyntaxAnalyzer()

corpus = pd.DataFrame(columns=['Original_news', 'Normalized_news', 'Class', 'Original_summary'])

for i in range(10000):

    doc_text = analyzer(xlsum['text'][i])
    doc_summary = analyzer(xlsum['summary'][i])
    
    normalized_text = analyzer.get_sentences(doc_text, normalize=True, upos=['NOUN', 'VERB', 'AUX', 'ADJ', 'X'])
    normalized_summary = analyzer.get_sentences(doc_summary, normalize=True, upos=['NOUN', 'VERB', 'AUX', 'ADJ', 'X'])

    result = processor(normalized_text, normalized_summary)

    if len(result) == 0:
        continue

    #ext_text = [text_for_print[i] for i in range(len(result)) if result[i] == 1]

    corpus.loc[len(corpus.index)] = [xlsum['text'][i], normalized_text, result, xlsum['summary'][i]]
    #print(f"Original text: {sentences_text}")
    #print(f"Extractive result: {ext_text}")
    #print(f"Original summary: {xlsum['summary'][i]}")

corpus.to_csv('./data_bertClass.csv')