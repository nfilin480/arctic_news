from transformers import MBartTokenizer, MBartForConditionalGeneration
import torch
#from decorators import trace


class Generator():

    #@trace
    def __init__(self) -> None:

        self.model_name = "IlyaGusev/mbart_ru_sum_gazeta"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = MBartTokenizer.from_pretrained(self.model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        
    #@trace
    def generate(self, article_text) -> str:

        # article_text = "С 1 сентября в России вступают в силу поправки в закон «О банкротстве» — теперь должники смогут освобождаться от непосильных обязательств во внесудебном порядке, если сумма задолженности составляет не менее 50 тыс. рублей и не превышает 500 тыс. рублей без учета штрафов, пени, процентов за просрочку платежа и прочих имущественных или финансовых санкций. У физлиц и индивидуальных предпринимателей появилась возможность пройти процедуру банкротства без участия суда и финансового управляющего — достаточно подать соответствующее заявление через МФЦ. Сумму задолженности и список всех известных заявителю кредиторов нужно предоставить самостоятельно. Если все условия соблюдены, сведения внесут в Единый федеральный реестр в течение трех рабочих дней. При этом на момент подачи заявления в отношении заявителя должно быть окончено исполнительное производство с возвращением исполнительного документа взыскателю. Это значит, что у потенциального банкрота не должно быть имущества, которое можно взыскать. Кроме того, в отношении гражданина не должно быть возбуждено другое исполнительное производство. В период всей процедуры заявитель не сможет брать займы, кредиты, выдавать поручительства, совершать иные обеспечительные сделки. Внесудебное банкротство будет длиться шесть месяцев, в течение которых также будет действовать мораторий на удовлетворение требований кредиторов, отмеченных в заявлении должника, и мораторий об уплате обязательных платежей. Кроме того, прекращается начисление неустоек и иных финансовых санкций; имущественные взыскания (кроме алиментов) также будут приостановлены. По завершению процедуры заявителя освободят от дальнейшего выполнения требований кредиторов, указанных в заявлении о признании его банкротом, а эта задолженность признается безнадежной. В прошлом месяце стало известно, что за первое полугодие 2020 года российские суды признали банкротами 42,7 тыс. граждан (в том числе индивидуальных предпринимателей) — по данным единого реестра «Федресурс», это на 47,2% больше показателя аналогичного периода 2019 года. Рост числа обанкротившихся граждан во втором квартале по сравнению с первым замедлился — такая динамика обусловлена тем, что в период ограничений с 19 марта по 11 мая суды редко рассматривали банкротные дела компаний и меньше, чем обычно, в отношении граждан, объяснял руководитель проекта «Федресурс» Алексей Юхнин. Он прогнозирует, что во втором полугодии мы увидим рост показателя, когда суды рассмотрят все дела, что не смогли ранее в режиме ограничений. По его данным, уже в июне число личных банкротств выросло до 11,5 тыс., что в два раза превышает показатель аналогичного периода 2019 года."
        input_ids = self.tokenizer(
            [article_text],
            max_length=600,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(self.device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            no_repeat_ngram_size=4
        )[0]
        summary = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return summary
    