import evaluate
import pandas as pd

#from decorator import trace

class Eval():

    #@trace
    def __init__(self) -> None:
        self.rouge_score = evaluate.load("rouge")
        self.bertscore = evaluate.load('bertscore')

    #@trace

    def rouge_scores(self, reference, generated):
        scores = self.rouge_score.compute(
            predictions=[generated], references=[reference]
        )


        return scores
    
    def bert_scores(self, reference, generated):
        scores = self.bertscore.compute(references=reference, predictions=generated, lang="ru")

        return scores
    
    def save_to_doc(self, scores):
        #TODO: сохранение сгенерированных саммари и оценок в файл 
        pass


#eval = Eval()
#data = pd.read_csv('E:\\IT_projects\\arctic_news\\datasets\\xlsum500.csv')

#metrics = eval(data['text'][0], data['summary'][0])
#print(metrics['rouge1'].mid)