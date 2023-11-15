import evaluate
from config import classInit

class Eval():

    @classInit
    def __init__(self) -> None:
        self.rouge_score = evaluate.load("rouge")

    @classInit
    def calc(self, generated, reference):
        scores = self.rouge_score.compute(
            predictions=[generated], references=[reference]
        )

        return scores
    
    def save_to_doc(self, scores):
        #TODO: сохранение сгенерированных саммари и оценок в файл 
        pass
