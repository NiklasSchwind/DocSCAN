from typing import List
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc


import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class DataAugmentation_new:
    def __init__(self, device: str, batch_size: int = 64):
        self.device = device
        self.batch_size = batch_size

    def ContextualWordEmbsAug(self, data: List[str]):
        aug = naw.ContextualWordEmbsAug()
        augmented_texts = aug.augment(data)
        return augmented_texts

    def SynonymAug(self, data: List[str]):
        aug = naw.SynonymAug()
        augmented_texts = aug.augment(data)
        return augmented_texts

    def AbstSummAug(self, data: List[str]):
        aug = naw.SynonymAug()
        augmented_texts = aug.augment(data)
        return augmented_texts

DataAugmentation = DataAugmentation_new(device = 'CUDA:1', batch_size = 64)

sentence= 'I am mister fox.'
print(DataAugmentation.SynonymAug([sentence]))
print(DataAugmentation.AbstSummAug([sentence]))
print(DataAugmentation.ContextualWordEmbsAug([sentence]))

