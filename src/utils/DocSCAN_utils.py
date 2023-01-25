import argparse
import os
import json
import pandas as pd
import torch
import random
import gc

from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
from transformers import MarianMTModel, MarianTokenizer

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import DistilBertTokenizerFast, DistilBertModel
from transformers import RobertaTokenizerFast, RobertaModel
from scipy.optimize import linear_sum_assignment

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from utils import *
from sklearn.metrics.pairwise import pairwise_distances

import matplotlib.pyplot as plt

#from visualizations import generate_word_clouds

class DocScanDataset(Dataset):
	def __init__(self, neighbor_df, embeddings, test_embeddings="", mode="train"):
		self.neighbor_df = neighbor_df
		self.embeddings = embeddings
		self.mode = mode
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		if mode == "train":
			self.examples = self.load_data()
		elif mode == "predict":
			self.examples = test_embeddings

	def load_data(self):
		examples = []
		for i,j in zip(self.neighbor_df["anchor"], self.neighbor_df["neighbor"]):
			examples.append((i,j))
		random.shuffle(examples)
		return examples

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, item):
		if self.mode == "train":
			anchor, neighbor = self.examples[item]
			sample = {"anchor": anchor, "neighbor": neighbor}
		elif self.mode == "predict":
			anchor = self.examples[item]
			sample = {"anchor": anchor}
		return sample
	def collate_fn(self, batch):
		anchors = torch.tensor([i["anchor"] for i in batch])
		out = self.embeddings[anchors].to(self.device)
		neighbors = torch.tensor([i["anchor"] for i in batch])
		out_2 = self.embeddings[neighbors].to(self.device)
		return {"anchor": out, "neighbor": out_2}

	def collate_fn_predict(self, batch):
		out = torch.vstack([i["anchor"] for i in batch]).to(self.device)
		return {"anchor": out}

class DocScanModel(torch.nn.Module):
	def __init__(self, num_labels, dropout, hidden_dim=768):
		super(DocScanModel, self).__init__()
		self.num_labels = num_labels
		self.classifier = torch.nn.Linear(hidden_dim, num_labels)
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		#self.device = "cpu"
		self.dropout = dropout

	def forward(self, feature):
		if self.dropout is not None:
			dropout = torch.nn.Dropout(p=self.dropout)
			feature = dropout(feature)
		output = self.classifier(feature)
		return output



class DocScanModel_new(torch.nn.Module):
	def __init__(self, num_labels, dropout, hidden_dim=768):
		super(DocScanModel_new, self).__init__()
		self.num_labels = num_labels
		self.classifier = torch.nn.Linear(hidden_dim, num_labels)
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		#self.device = "cpu"
		self.dropout = dropout

	def forward(self, feature):
		if self.dropout is not None:
			dropout = torch.nn.Dropout(p=self.dropout)
			feature = dropout(feature)
		output = self.classifier(feature)
		return output


class Backtranslation:
	def __init__(self, batch_size ):
		self.batch_size = batch_size
		self.tokenizer_fr_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
		self.tokenizer_en_fr = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
		self.model_fr_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
		self.model_en_fr = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
		self.inbetween_language = 'fr'

	def format_batch_texts(self, language_code, batch_texts):

		formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]

		return formated_bach

	def perform_translation(self, batch_texts, model, tokenizer, language="fr"):
		# Prepare the text data into appropriate format for the model
		formated_batch_texts = self.format_batch_texts(language, batch_texts)

		# Generate translation using model
		translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))

		# Convert the generated tokens indices back into text
		translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

		return translated_texts

	def combine_texts(self, original_texts, back_translated_batch):

		return set(original_texts + back_translated_batch)

	def perform_back_translation_with_augmentation(self, batch_texts, original_language="en", temporary_language="fr"):

		# Translate from Original to Temporary Language
		tmp_translated_batch = self.perform_translation(batch_texts, self.model_en_fr, self.tokenizer_en_fr, temporary_language)

		# Translate Back to English
		back_translated_batch = self.perform_translation(tmp_translated_batch, self.model_fr_en, self.tokenizer_fr_en, original_language)

		# Return The Final Result
		return back_translated_batch

	def divide_chunks(self,list, number):
		# looping till length l
		for i in range(0, len(list), number):
			yield list[i:i + number]

	def backtranslate(self,texts):
		augmented_texts = []
		for batch in self.divide_chunks(texts,self.batch_size):
			augmented_texts.append(self.perform_back_translation_with_augmentation(batch))

		return augmented_texts







