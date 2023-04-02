from typing import List
from transformers import MarianMTModel, MarianTokenizer

class DataAugmentation:
    def __init__(self, device: str, batch_size: int = 128):
        self.device = device
        self.batch_size = batch_size


    def Backtranslation(self, data: List[str], translate_languages: List[str], original_language : str = 'en'):

        translation_order =  translate_languages + list(reversed(translate_languages)) + [original_language]
        language_before = original_language
        for language in translation_order:
            augmented_data = []
            tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{language_before}-{language}').to(
                self.device)
            model = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{language_before}-{language}').to(self.device)
            for batch in self.divide_chunks(data, self.batch_size):
                augmented_data.append(self._translate_texts(tokenizer,model,batch))

        return data

    def divide_chunks(self, list, number):
        # looping till length l
        for i in range(0, len(list), number):
            yield list[i:i + number]


    def _translate_texts(self,tokenizer,model,texts: List[str]):

        augmented_texts = []

        mo

        return augmented_texts

