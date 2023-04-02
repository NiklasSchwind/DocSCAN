from typing import List

class DataAugmentation:
    def __init__(self, text: List[str]):
        self.text = text

    def Backtranslation(self, languages: List[str], orig_language : str = 'en'):