import eval.eval
import generator.bart
from decorator import trace


@trace
def run(gen, evaluate):
    while True:
        reference = input('text: ')

        if len(reference) < 3:
            break

        try:
            generated_summary = gen.generate(reference)
            print(generated_summary)
            scores = evaluate.calc(reference, generated_summary)
            print(scores)
        except ValueError:
            print("При генерации возникла ошибка")


if __name__ == '__main__':
    gen = generator.bart.Generator()
    evaluate = eval.eval.Eval()
    run(gen, evaluate)


