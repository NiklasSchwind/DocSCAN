import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from typing import List
import nltk
from nltk.corpus import stopwords
import multiprocessing








class Doc2Vec_Embedder:
    def __init__(self, train: List[str], test: List[str]):
        self.train = pd.DataFrame(train, columns = ['text'])
        self.test = pd.DataFrame(test, columns = ['text'])
        self.cores = multiprocessing.cpu_count()

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
        train = self.train['text'].apply(self.clean_text)
        test = self.test['text'].apply(self.clean_text)
        train_tagged = train.apply(lambda r: TaggedDocument(words=self.tokenize_text(r['text']),axis=1))
        test_tagged = test.apply(lambda r: TaggedDocument(words=self.tokenize_text(r['text']), axis=1))
        model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers= self.cores)
        model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])
        for epoch in range(30):
            model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]),
                             total_examples=len(train_tagged.values), epochs=1)
            model_dbow.alpha -= 0.002
            model_dbow.min_alpha = model_dbow.alpha



