import sys, os, json, argparse
import pandas as pd
from sentence_transformers import SentenceTransformer,InputExample, models, losses, util, datasets, evaluation
from utils.memory import MemoryBank
import torch
from utils.DocSCAN_utils import DocScanDataset, DocScanModel, Backtranslation
from utils.losses import SCANLoss, ConfidenceBasedCE
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from scipy.special import softmax
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from utils.utils import *
import random
import nltk
from utils.EncodeDropout import encode_with_dropout
from transformers import MarianMTModel, MarianTokenizer
from TrainingWithPrototypes import Dataset_Bert, BertClassifier, finetune_BERT,evaluate_Bert, softmax
nltk.download('punkt')

def evaluate(targets, predictions, verbose=0, mode = 'test'):
	# right, do this...
	# shouldn't be too hard
	hungarian_match_metrics = hungarian_evaluate(targets, predictions)
	cm = hungarian_match_metrics["confusion matrix"]
	clf_report = hungarian_match_metrics["classification_report"]
	#print (fn_val, hungarian_match_metrics)
	del hungarian_match_metrics["classification_report"]
	del hungarian_match_metrics["confusion matrix"]
	if verbose:
		print (cm, "\n", clf_report)
		print (hungarian_match_metrics)
	print (f"ACCURACY {mode} ", np.round(hungarian_match_metrics["ACC"], 3))
	return hungarian_match_metrics


class DocSCANPipeline():
	def __init__(self, args):
		self.args = args
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		os.makedirs(self.args.path, exist_ok=True)

	def load_data(self, filename):
		sentences, labels = [], []
		with open(filename) as f:
			for line in f:
				line = json.loads(line)
				sentences.append(line["text"])
				labels.append(line["label"])
		df = pd.DataFrame(list(zip(sentences, labels)), columns=["sentence", "label"])
		return df
			
	def embedd_sentences(self, sentences):
		embedder = SentenceTransformer(self.args.sbert_model)
		embedder.max_seq_length = self.args.max_seq_length
		corpus_embeddings = embedder.encode(sentences, batch_size=32, show_progress_bar=True)
		return corpus_embeddings

	def embedd_sentences_method(self, sentences, method, loadpath = None):
		if method == 'SBert':
			embedder = SentenceTransformer(self.args.sbert_model)
			embedder.max_seq_length = self.args.max_seq_length
			corpus_embeddings = embedder.encode(sentences, batch_size=32, show_progress_bar=True)

		elif method == 'SBert_dropout':
			with torch.no_grad():
				embedder = SentenceTransformer(self.args.sbert_model)
				embedder.max_seq_length = self.args.max_seq_length
				embedder.train()
				corpus_embeddings = encode_with_dropout(embedder, sentences, batch_size=32, show_progress_bar=True, eval = False)
				embedder.eval()

		elif method == 'SimCSE':
			if loadpath is None:
				# Define sentence transformer model using CLS pooling
				model_name = 'distilroberta-base' #'sentence-transformers/all-mpnet-base-v2'#'distilroberta-base'
				word_embedding_model = models.Transformer(model_name, max_seq_length=32)
				pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
				SimCSEmodel = SentenceTransformer(modules=[word_embedding_model, pooling_model])

				# Create sentence pairs for training
				TrainData_paired = [InputExample(texts=[s, s]) for s in sentences]

				# DataLoader to batch the data using recommended batchsize
				TrainData_batched = torch.utils.data.DataLoader(TrainData_paired, batch_size=128, shuffle=True)

				# Define recommended loss function
				train_loss = losses.MultipleNegativesRankingLoss(SimCSEmodel)

				# Call the fit method
				SimCSEmodel.fit(
					train_objectives=[(TrainData_batched, train_loss)],
					epochs=5,
					show_progress_bar=True
				)

				SimCSEmodel.save('output/simcse-model') #TODO refine
			else:
				SimCSEmodel = torch.load(loadpath)

			# get wanted embeddings
			corpus_embeddings = SimCSEmodel.encode(sentences)

		elif method == 'TSDAE':
			if loadpath is None:
				# Define your sentence transformer model using CLS pooling
				model_name = 'bert-base-uncased'
				word_embedding_model = models.Transformer(model_name)
				pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
				TSDAEModel = SentenceTransformer(modules=[word_embedding_model, pooling_model])

				# Transform dataset to right format
				train_dataset = datasets.DenoisingAutoEncoderDataset(sentences)

				# DataLoader to batch data, use recommended batch size
				train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

				# Define recommanded loss function
				train_loss = losses.DenoisingAutoEncoderLoss(TSDAEModel, decoder_name_or_path=model_name,
															 tie_encoder_decoder=True)

				# Call the fit method
				TSDAEModel.fit(
					train_objectives=[(train_dataloader, train_loss)],
					epochs=1,
					weight_decay=0,
					scheduler='constantlr',
					optimizer_params={'lr': 3e-5},
					show_progress_bar=True
				)

				TSDAEModel.save('output/tsdae-model')

			else:
				TSDAEModel = torch.load(loadpath)
			# wanted word embeddings

			corpus_embeddings = TSDAEModel.encode(sentences)


		return corpus_embeddings

	def create_neighbor_dataset(self, indices=None):
		if indices is None:
			indices = self.memory_bank.mine_nearest_neighbors(self.args.num_neighbors, show_eval=False, calculate_accuracy=False)
		examples = []
		for i, index in enumerate(indices): 
			anchor = i
			neighbors = index
			#print (len(neighbors))
			for neighbor in neighbors:
				if neighbor == i:
					continue
				examples.append((anchor, neighbor))
		df = pd.DataFrame(examples, columns=["anchor", "neighbor"])
		if self.args.num_neighbors == 5:
			df.to_csv(os.path.join(self.args.path, "neighbor_dataset.csv"))
		else:
			df.to_csv(os.path.join(self.args.path, "neighbor_dataset" + str(self.args.num_neighbors) + ".csv"))
		#sys.exit(0)
		#sys.exit(0)
		return df

	def retrieve_neighbours_gpu(self, X, batchsize=16384, num_neighbors=5):
		import faiss
		res = faiss.StandardGpuResources()  # use a single GPU
		n, dim = X.shape[0], X.shape[1]
		index = faiss.IndexFlatIP(dim) # create CPU index
		gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index) # create GPU index
		gpu_index_flat.add(X)         # add vectors to the index

		all_indices = []
		for i in tqdm(range(0, n, batchsize)):
			features = X[i:i + batchsize]
			distances, indices = gpu_index_flat.search(features, num_neighbors)
			all_indices.extend(indices)
		return all_indices

	def get_predictions(self, model, dataloader):
		predictions, probs = [], []
		epoch_iterator = tqdm(dataloader, total=len(dataloader))
		model.eval()
		print(len(dataloader))
		with torch.no_grad():
			for i, batch in enumerate(epoch_iterator):
				model.eval()
				output_i = model(batch["anchor"])
				#probs.extend(torch.nn.functional.softmax(output_i, dim=1).cpu().numpy())
				probs.extend(output_i.cpu().numpy())
				predictions.extend(torch.argmax(output_i, dim=1).cpu().numpy())
		print(len(predictions))
		return predictions, probs


	def train(self, model, optimizer, criterion, train_dataloader, num_classes):
		train_iterator = range(int(self.args.num_epochs))
		cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
		softmax = torch.nn.Softmax()
		# train

		targets_map = {i:j for j,i in enumerate(np.unique(self.df_test["label"]))}
		targets = [targets_map[i] for i in self.df_test["label"]]

		for epoch in train_iterator:
			bar_desc = "Epoch %d of %d | num classes %d | Iteration" % (epoch + 1, len(train_iterator), num_classes)
			epoch_iterator = tqdm(train_dataloader, desc=bar_desc)
			for step, batch in enumerate(epoch_iterator):
				anchor, neighbor = batch["anchor"], batch["neighbor"]
				anchors_output, neighbors_output = model(anchor), model(neighbor)
				
				total_loss, consistency_loss, entropy_loss = criterion(anchors_output, neighbors_output)
				total_loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				model.zero_grad()
			#predictions, probabilities = self.get_predictions(model, predict_dataloader)
			#evaluate(np.array(targets), np.array(predictions),verbose=0)


		optimizer.zero_grad()
		model.zero_grad()
		return model


	def train_model(self):
		train_dataset = DocScanDataset(self.neighbor_dataset, self.X, mode="train")
		model = DocScanModel(self.args.num_classes, self.args.dropout).to(self.device)
		optimizer = torch.optim.Adam(model.parameters())
		criterion = SCANLoss()
		criterion.to(self.device)

		batch_size = self.args.batch_size
		train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn = train_dataset.collate_fn, batch_size=batch_size)
		# train
		model = self.train(model, optimizer, criterion, train_dataloader, self.args.num_classes)

		return model






	def augment(self, sentences, method):

		if method == 'backtranslation':
			Backtranslator = Backtranslation(batch_size = 128)
			sentences_augmented = Backtranslator.backtranslate(sentences)
			df_train_augmented = pd.DataFrame(list(zip(sentences_augmented, list(sentences['label']))),
											  columns=["sentence", "label"])
		else:
			df_train_augmented = sentences


		return df_train_augmented


	def fine_tune_through_selflabeling(self, df_train, model, augmentation):

		# train data
		predict_dataset_train = DocScanDataset(self.neighbor_dataset, self.X, mode="predict",
											   test_embeddings=self.X)
		predict_dataloader_train = torch.utils.data.DataLoader(predict_dataset_train, shuffle=False,
															   collate_fn=predict_dataset_train.collate_fn_predict,
															   batch_size=self.args.batch_size)

		predictions_train, probabilities_train = self.get_predictions(model, predict_dataloader_train)
		targets_map_train = {i: j for j, i in enumerate(np.unique(df_train["label"]))}
		targets_train = [targets_map_train[i] for i in df_train["label"]]
		print(len(targets_train), len(predictions_train))
		evaluate(np.array(targets_train), np.array(predictions_train), mode = 'train')

		docscan_clusters_train = evaluate(np.array(targets_train), np.array(predictions_train),mode = 'train')["reordered_preds"]
		df_train["label"] = targets_train
		df_train["clusters"] = docscan_clusters_train
		df_train["probabilities"] = probabilities_train

		df_augmented = self.augment(df_train, method=augmentation)

		embeddings_augmented = self.embedd_sentences_method(df_augmented['sentence'], method='SBert_dropout')
		embeddings_augmented = torch.from_numpy(embeddings_augmented)
		# augmented data
		predict_dataset_augmented = DocScanDataset(self.neighbor_dataset, embeddings_augmented, mode="predict",
											   test_embeddings=embeddings_augmented)

		predict_dataloader_augmented = torch.utils.data.DataLoader(predict_dataset_augmented, shuffle=False,
															   collate_fn=predict_dataset_augmented.collate_fn_predict,
															   batch_size=self.args.batch_size)

		predictions_augmented, probabilities_augmented = self.get_predictions(model, predict_dataloader_augmented)
		targets_map_augmented = {i: j for j, i in enumerate(np.unique(df_augmented["label"]))}
		targets_augmented = [targets_map_augmented[i] for i in df_augmented["label"]]
		print(len(targets_augmented), len(predictions_augmented))
		evaluate(np.array(targets_augmented), np.array(predictions_augmented), mode = 'augmented')

		docscan_clusters_augmented = evaluate(np.array(targets_augmented), np.array(predictions_augmented), mode= 'augmented')["reordered_preds"]
		df_augmented["label"] = targets_augmented
		df_augmented["clusters"] = docscan_clusters_augmented
		df_augmented["probabilities"] = probabilities_augmented

		optimizer = torch.optim.Adam(model.parameters())
		criterion = ConfidenceBasedCE(threshold=0.99, apply_class_balancing=True)
		criterion.to(self.device)

		batch_size = self.args.batch_size

		dataset = list(zip(self.X, embeddings_augmented))
		dataloader = torch.utils.data.DataLoader(dataset , shuffle=True,batch_size=self.args.batch_size)

		train_iterator = range(int(self.args.num_epochs))

		targets_map_augmented = {i: j for j, i in enumerate(np.unique(df_augmented["label"]))}
		targets_augmented = [targets_map_augmented[i] for i in df_augmented["label"]]

		for epoch in train_iterator:
			bar_desc = "Epoch %d of %d | num classes %d | Iteration" % (epoch + 1, len(train_iterator), self.args.num_classes)
			epoch_iterator = tqdm(dataloader, desc=bar_desc)
			for step, batch in enumerate(epoch_iterator):
				try:
					anchor_weak, anchor_strong = batch[0], batch[1]
					original_output, augmented_output = model(anchor_weak), model(anchor_strong)
					total_loss = criterion(original_output, augmented_output)
					total_loss.backward()
					optimizer.step()
					optimizer.zero_grad()
					model.zero_grad()
				except ValueError:
					print(f'Recieved Value Error in step {step}')

		predictions, probabilities = self.get_predictions(model, predict_dataloader_train)
		evaluate(np.array(targets_train), np.array(predictions),verbose=0)
		optimizer.zero_grad()
		model.zero_grad()
		return model


	def run_main(self):
		# embedd using SBERT

		print ("loading data...")
		train_file = os.path.join(self.args.path, "train.jsonl")
		predict_file = os.path.join(self.args.path, "test.jsonl")

		df_train = self.load_data(train_file)
		args.num_classes = df_train.label.nunique()
		self.df_test = self.load_data(predict_file)


		print ("embedding sentences...")
		if os.path.exists(os.path.join(self.args.path, "embeddings.npy")):
			self.embeddings = np.load(os.path.join(self.args.path, "embeddings.npy"))
		else:
			self.embeddings = self.embedd_sentences_method(df_train["sentence"], 'SBert') #self.embeddings = self.embedd_sentences(df_train["sentence"])
			np.save(os.path.join(self.args.path, "embeddings"), self.embeddings)

		# torch tensor of embeddings
		self.X = torch.from_numpy(self.embeddings)
		if os.path.exists(os.path.join(self.args.path, "embeddings_test.npy")):
			self.embeddings_test = np.load(os.path.join(self.args.path, "embeddings_test.npy"))
		else:
			self.embeddings_test = self.embedd_sentences_method(self.df_test["sentence"], 'SBert')# self.embedd_sentences(self.df_test["sentence"])
			np.save(os.path.join(self.args.path, "embeddings_test"), self.embeddings_test)

		self.X_test = torch.from_numpy(self.embeddings_test)

		print ("retrieving neighbors...")

		if os.path.exists(os.path.join(self.args.path, "neighbor_dataset.csv")) and self.args.num_neighbors == 5:
			print ("loading neighbor dataset")
			self.neighbor_dataset = pd.read_csv(os.path.join(self.args.path, "neighbor_dataset.csv"))
		elif os.path.exists(os.path.join(self.args.path, "neighbor_dataset" + str(self.args.num_neighbors) + ".csv")):
			self.neighbor_dataset = pd.read_csv(os.path.join(self.args.path, "neighbor_dataset" + str(self.args.num_neighbors) + ".csv"))
		else:
			if self.device == "cpu":
				self.memory_bank = MemoryBank(self.X, "", len(self.X), 
						        self.X.shape[-1],
						        self.args.num_classes)
				self.neighbor_dataset = self.create_neighbor_dataset()
			else:
				indices = self.retrieve_neighbours_gpu(self.X.numpy(), num_neighbors = self.args.num_neighbors)
				self.neighbor_dataset = self.create_neighbor_dataset(indices=indices)

		results = []
		results_after_selflabeling = []

		targets_map = {i:j for j,i in enumerate(np.unique(self.df_test["label"]))}
		targets = [targets_map[i] for i in self.df_test["label"]]


		for _ in range(10):
			model = self.train_model()
			# test data
			predict_dataset = DocScanDataset(self.neighbor_dataset, self.X_test, mode="predict",
											 test_embeddings=self.X_test)
			predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
															 collate_fn=predict_dataset.collate_fn_predict,
															 batch_size=self.args.batch_size)
			predictions, probabilities = self.get_predictions(model, predict_dataloader)


			print("docscan trained with n=", self.args.num_classes, "clusters...")

			targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
			targets = [targets_map[i] for i in self.df_test["label"]]
			print(len(targets), len(predictions))
			evaluate(np.array(targets), np.array(predictions))

			docscan_clusters = evaluate(np.array(targets), np.array(predictions))["reordered_preds"]
			self.df_test["label"] = targets
			self.df_test["clusters"] = docscan_clusters
			self.df_test["probabilities"] = probabilities
			acc_test = np.mean(self.df_test["label"] == self.df_test["clusters"])
			results.append(acc_test)

			model = self.fine_tune_through_selflabeling(df_train, model, augmentation = 'backtranslation')

			predictions, probabilities = self.get_predictions(model, predict_dataloader)

			print("docscan trained with n=", self.args.num_classes, "clusters...")

			targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
			targets = [targets_map[i] for i in self.df_test["label"]]
			print(len(targets), len(predictions))
			evaluate(np.array(targets), np.array(predictions), mode = 'after selflabeling')

			docscan_clusters = evaluate(np.array(targets), np.array(predictions),mode = 'after selflabeling')["reordered_preds"]
			self.df_test["label"] = targets
			self.df_test["clusters"] = docscan_clusters
			self.df_test["probabilities"] = probabilities
			acc_test = np.mean(self.df_test["label"] == self.df_test["clusters"])

			results_after_selflabeling.append(acc_test)







			"""
			model = self.train_model()
			# test data
			predict_dataset = DocScanDataset(self.neighbor_dataset, self.X_test, mode="predict", test_embeddings=self.X_test)
			predict_train = DocScanDataset(self.neighbor_dataset, self.X, mode="predict", test_embeddings=self.X)
			predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False, collate_fn = predict_dataset.collate_fn_predict, batch_size=self.args.batch_size)
			predict_train_dataloader = torch.utils.data.DataLoader(predict_train, shuffle=False,
															 collate_fn=predict_train.collate_fn_predict,
															 batch_size=self.args.batch_size)

			predictions, probabilities = self.get_predictions(model, predict_dataloader)
			# train data
			predictions_train, probabilities_train = self.get_predictions(model, predict_train_dataloader)

			print ("docscan trained with n=", self.args.num_classes, "clusters...")
 

			targets_map = {i:j for j,i in enumerate(np.unique(self.df_test["label"]))}
			targets = [targets_map[i] for i in self.df_test["label"]]
			targets_map_train = {i: j for j, i in enumerate(np.unique(df_train["label"]))}
			targets_train = [targets_map[i] for i in df_train["label"]]
			print (len(targets), len(predictions))
			evaluate(np.array(targets), np.array(predictions))
			print(len(targets_train), len(predictions_train))
			evaluate(np.array(targets_train), np.array(predictions_train), mode = 'test')

			docscan_clusters = evaluate(np.array(targets), np.array(predictions))["reordered_preds"]
			docscan_clusters_train = evaluate(np.array(targets_train), np.array(predictions_train),mode = 'test')["reordered_preds"]
			self.df_test["label"] = targets
			self.df_test["clusters"] = docscan_clusters
			df_train["label"] = targets_train
			df_train["clusters"] = docscan_clusters_train

			self.df_test["probabilities"] = np.amax(np.array(torch.nn.Softmax(dim=1)(torch.tensor(probabilities)).tolist()),axis = 1)
			df_train["probabilities"] = np.amax(np.array(torch.nn.Softmax(dim=1)(torch.tensor(probabilities_train)).tolist()), axis=1)
			acc_test = np.mean(self.df_test["label"] == self.df_test["clusters"])
			acc_train = np.mean(df_train["label"] == df_train["clusters"])
			results.append(acc_test)
			print('Test Probabilities:')
			print(self.df_test["probabilities"])

			lookup = {}
			for i,label in enumerate(self.df_test['label']):
				if self.df_test['probabilities'][i] >= 0.99 and label in lookup.keys():
					lookup[label] += 1
				elif self.df_test['probabilities'][i] >= 0.99:
					lookup[label] = 1

			print(lookup)

			print('Train Probabilities:')
			print(df_train["probabilities"])

			lookup_train = {}
			for i, label in enumerate(df_train['label']):
				if df_train['probabilities'][i] >= 0.99 and label in lookup_train.keys():
					lookup_train[label] += 1
				elif df_train['probabilities'][i] >= 0.99:
					lookup_train[label] = 1

			print(lookup_train)
			"""
		print("mean accuracy test", np.mean(results).round(3), "(" + str(np.std(results).round(3)) + ")")

		print("mean accuracy test after selflabeling", np.mean(results_after_selflabeling).round(3), "(" + str(np.std(results_after_selflabeling).round(3)) + ")")


	def run_main_train_extra_model_with_prototypes(self):
		# embedd using SBERT

		print("loading data...")
		train_file = os.path.join(self.args.path, "train.jsonl")
		predict_file = os.path.join(self.args.path, "test.jsonl")

		df_train = self.load_data(train_file)
		args.num_classes = df_train.label.nunique()
		self.df_test = self.load_data(predict_file)

		print("embedding sentences...")
		if os.path.exists(os.path.join(self.args.path, "embeddings.npy")):
			self.embeddings = np.load(os.path.join(self.args.path, "embeddings.npy"))
		else:
			self.embeddings = self.embedd_sentences(df_train["sentence"])
			np.save(os.path.join(self.args.path, "embeddings"), self.embeddings)

		# torch tensor of embeddings
		self.X = torch.from_numpy(self.embeddings)
		if os.path.exists(os.path.join(self.args.path, "embeddings_test.npy")):
			self.embeddings_test = np.load(os.path.join(self.args.path, "embeddings_test.npy"))
		else:
			self.embeddings_test = self.embedd_sentences(self.df_test["sentence"])
			np.save(os.path.join(self.args.path, "embeddings_test"), self.embeddings_test)

		self.X_test = torch.from_numpy(self.embeddings_test)

		print("retrieving neighbors...")

		if os.path.exists(os.path.join(self.args.path, "neighbor_dataset.csv")) and self.args.num_neighbors == 5:
			print("loading neighbor dataset")
			self.neighbor_dataset = pd.read_csv(os.path.join(self.args.path, "neighbor_dataset.csv"))
		elif os.path.exists(os.path.join(self.args.path, "neighbor_dataset" + str(self.args.num_neighbors) + ".csv")):
			self.neighbor_dataset = pd.read_csv(
				os.path.join(self.args.path, "neighbor_dataset" + str(self.args.num_neighbors) + ".csv"))
		else:
			if self.device == "cpu":
				self.memory_bank = MemoryBank(self.X, "", len(self.X),
											  self.X.shape[-1],
											  self.args.num_classes)
				self.neighbor_dataset = self.create_neighbor_dataset()
			else:
				indices = self.retrieve_neighbours_gpu(self.X.numpy(), num_neighbors=self.args.num_neighbors)
				self.neighbor_dataset = self.create_neighbor_dataset(indices=indices)

		results = []
		results_extra = []

		targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
		targets = [targets_map[i] for i in self.df_test["label"]]

		for _ in range(10):
			model = self.train_model()
			# test data
			predict_dataset = DocScanDataset(self.neighbor_dataset, self.X_test, mode="predict",
											 test_embeddings=self.X_test)
			predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
															 collate_fn=predict_dataset.collate_fn_predict,
															 batch_size=self.args.batch_size)
			predictions, probabilities = self.get_predictions(model, predict_dataloader)
			# train data

			print("docscan trained with n=", self.args.num_classes, "clusters...")

			targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
			targets = [targets_map[i] for i in self.df_test["label"]]
			print(len(targets), len(predictions))
			evaluate(np.array(targets), np.array(predictions))

			docscan_clusters = evaluate(np.array(targets), np.array(predictions))["reordered_preds"]
			self.df_test["label"] = targets
			self.df_test["clusters"] = docscan_clusters
			self.df_test["probabilities"] = probabilities
			acc_test = np.mean(self.df_test["label"] == self.df_test["clusters"])
			results.append(acc_test)

			predict_dataset_train = DocScanDataset(self.neighbor_dataset, self.X, mode="predict",
												   test_embeddings=self.X)
			predict_dataloader_train = torch.utils.data.DataLoader(predict_dataset_train, shuffle=False,
																   collate_fn=predict_dataset_train.collate_fn_predict,
																   batch_size=self.args.batch_size)

			predictions_train, probabilities_train = self.get_predictions(model, predict_dataloader_train)
			targets_map_train = {i: j for j, i in enumerate(np.unique(df_train["label"]))}
			targets_train = [targets_map_train[i] for i in df_train["label"]]
			print(len(targets_train), len(predictions_train))
			evaluate(np.array(targets_train), np.array(predictions_train), mode='train')

			docscan_clusters_train = evaluate(np.array(targets_train), np.array(predictions_train), mode='train')[
				"reordered_preds"]
			df_train["label"] = targets_train
			df_train["clusters"] = docscan_clusters_train
			df_train["probabilities"] = probabilities_train

			df_ExtraModel = df_train[df_train["probabilities"].apply(softmax).apply(np.max) >= 0.99]
			df_ExtraModel = df_ExtraModel[['sentence','clusters']].rename({'sentence':'text', 'clusters': 'cluster'},axis='columns')

			print(df_ExtraModel )
			Extra_Model = BertClassifier()
			finetune_BERT(Extra_Model, df_ExtraModel, 1e-6, 5)

			df_ExtraModel_test = self.df_test[self.df_test["probabilities"].apply(softmax).apply(np.max) >= 0.99]
			df_ExtraModel_test = df_ExtraModel_test[['sentence', 'clusters']].rename({'sentence': 'text', 'clusters': 'cluster'},
																		   axis='columns')



			acc_extramodel = evaluate_Bert(Extra_Model, df_ExtraModel_test)
			results_extra.append(acc_extramodel)





		print("mean accuracy", np.mean(results).round(3), "(" + str(np.std(results).round(3)) + ")")
		print("mean accuracy extra model", np.mean(results_extra).round(3), "(" + str(np.std(results_extra).round(3)) + ")")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", type=str, help="path to output path where output of docscan gets saved")
	parser.add_argument("--sbert_model", default="sentence-transformers/all-mpnet-base-v2", type=str, help="SBERT model to use to embedd sentences") 
	parser.add_argument("--max_seq_length", default=128, type=int, help="max seq length of sbert model, sequences longer than this get truncated at this value")
	parser.add_argument("--topk", type=int, default=5, help="numbers of neighbors retrieved to build SCAN training set")
	parser.add_argument("--num_classes", type=int, default=10, help="numbers of clusters")
	parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.")
	parser.add_argument("--dropout", default=0.1, type=float, help="dropout for DocSCAN model")
	parser.add_argument("--num_epochs", default=5, type=int, help="number of epochs to train DocSCAN model")
	parser.add_argument("--num_neighbors", default=5, type=int, help="number of epochs to train DocSCAN model")
	args = parser.parse_args()

	if args.dropout == 0:
		args.dropout = None
	docscan = DocSCANPipeline(args)
	docscan.run_main_train_extra_model_with_prototypes()

