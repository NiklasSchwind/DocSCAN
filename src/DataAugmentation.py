from typing import List

class DataAugmentation:
    def __init__(self, text: List[str],  device: str, batch_size: int = 128):
        self.text = text
        self.device = device
        self.batch_size = batch_size
    def Backtranslation(self, languages: List[str], original_language : str = 'en'):
        texts = self.text
        tokenizers = {}
        translation_models = {}
        language_before = original_language
        for language in languages:
            tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{language_before}-{language}').to(self.device)
            model = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{language_before}-{language}').to(self.device)
            texts = self._translate_texts(tokenizer,model,texts)
        for language in reversed(languages):


    def _translate_texts(self,tokenizer,model,texts: List[str]):


