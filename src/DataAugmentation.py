from typing import List
from transformers import MarianMTModel, MarianTokenizer
import random

class DataAugmentation:
    def __init__(self, device: str, batch_size: int = 128):
        self.device = device
        self.batch_size = batch_size


    def Backtranslation(self, data: List[str], language_order: List[str], original_language : str = 'en'):

        language_before = original_language
        for language in language_order:
            augmented_data = []
            tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{language_before}-{language}').to(
                self.device)
            model = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{language_before}-{language}').to(self.device)
            for batch in self._divide_chunks(data, self.batch_size):
                batch.to(self.device)
                augmented_data.append(self._translate_texts(tokenizer,model,language_before, batch))
            data = augmented_data
            language_before = language

        return data

    def _divide_chunks(self, list, number):
        # looping till length l
        for i in range(0, len(list), number):
            yield list[i:i + number]

    def _format_batch_texts(self, language:str, batch_texts:List[str]):

        formated_bach = [">>{}<< {}".format(language, text) for text in batch_texts]

        return formated_bach


    def _translate_texts(self,tokenizer,model,language: str,texts: List[str]):

        formated_batch_texts = self._format_batch_texts(language, texts)

        # Generate translation using model
        translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))

        # Convert the generated tokens indices back into text
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        return translated_texts

    def Random_deletion(self, texts : List[str], ratio : float):

        return [' '.join([word for word in text.split(' ') if random.random() >= ratio]) for text in texts]



