import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
import re
from bs4 import BeautifulSoup
from typing import List
import nltk
import multiprocessing
import json
import os







class Doc2Vec_Embedder:
    def __init__(self, train: List[str]):

        self.train = pd.DataFrame(train, columns = ['text'])
        self.cores = multiprocessing.cpu_count()
        self.model = None

    def clean_text(self, text):

        text = BeautifulSoup(text, "lxml").text
        text = re.sub(r'\|\|\|', r' ', text)
        text = re.sub(r'http\S+', r'<URL>', text)
        text = text.lower()
        text = text.replace('x', '')
        return text

    def tokenize_text(self, text):

        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) < 2:
                    continue
                tokens.append(word.lower())
        return tokens

    def TrainDoc2Vec(self):


        train_tagged = [TaggedDocument(self.tokenize_text(doc), [i]) for i, doc in enumerate(self.train['text'])] #train.apply(lambda r: TaggedDocument(words=self.tokenize_text(r['text']), tags=[1], axis=1))
        model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers= self.cores)
        model_dbow.build_vocab([x for x in tqdm(train_tagged)])


        for epoch in range(30):
            model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged)]),
                             total_examples=len(train_tagged), epochs=1)
            model_dbow.alpha -= 0.002
            model_dbow.min_alpha = model_dbow.alpha

        return model_dbow

    def embed(self, data: List[str]):

        if self.model is None:
            self.model = self.TrainDoc2Vec()

        data = pd.DataFrame(data, columns = ['text'])
        data = data['text'].apply(self.clean_text)
        data_tagged = [TaggedDocument(self.tokenize_text(doc), [i]) for i, doc in enumerate(data['text'])]
        embeddings = [self.model.infer_vector(doc.words, steps=20) for doc in data_tagged ]

        return embeddings






def load_data(filename):
    sentences, labels = [], []
    with open(filename) as f:
        for line in f:
            line = json.loads(line)
            sentences.append(line["text"])
            labels.append(line["label"])
    df = pd.DataFrame(list(zip(sentences, labels)), columns=["sentence", "label"])
    return df


def create_embeddings(dataset_folder: str):
    train_data = os.path.join( dataset_folder, "train.jsonl")
    test_data = os.path.join( dataset_folder, "test.jsonl")
    df_train = load_data(train_data)

    df_test = load_data(test_data)
    train = df_train.sentence.values.tolist()
    test = df_test.sentence.values.tolist()
    print(train)
    Model = Doc2Vec_Embedder(train = train)
    embeddings_train = np.array(Model.embed(train))
    embeddings_test = np.array(Model.embed(test))
    np.save(os.path.join(dataset_folder, f"train-{Doc2Vec}-embeddings.npy"), embeddings_train)
    np.save(os.path.join(dataset_folder, f"test-{Doc2Vec}-embeddings.npy"), embeddings_test)

if __name__ == '__main__':
    create_embeddings('/vol/fob-vol7/mi19/schwindn/DocSCAN/TREC-6')