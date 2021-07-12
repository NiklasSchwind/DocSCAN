import argparse
import os
import json
import pandas as pd
import torch
import random
import gc

from losses import SCANLoss
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import DistilBertTokenizerFast, DistilBertModel
from transformers import RobertaTokenizerFast, RobertaModel
from scipy.optimize import linear_sum_assignment

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from utils import *
from sklearn.metrics.pairwise import pairwise_distances

from kneelocator import KneeLocator
import matplotlib.pyplot as plt

class ScanDataset(Dataset):
	def __init__(self, filename, embeddings=None, mode="train", translation_fn=None):
		self.filename = filename
		self.mode = mode
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		#self.device = "cpu"
		self.translation_fn= translation_fn
		if mode == "train":
			self.examples = self.load_data()
		elif mode == "val":
			self.examples = self.load_val_data()
		self.embeddings = embeddings
	def load_data(self):
		examples = []
		df = pd.read_pickle(self.filename)
		anchors = df["anchor"].tolist()
		neighbours = df["neighbour"].tolist()
		labels = df["label"].tolist()
		counts = defaultdict(int)
		for i,j in zip(anchors, neighbours):
			examples.append((i,j))

		random.shuffle(examples)
		return examples


	def load_val_data(self):
		df = pd.read_pickle(self.filename)
		labels=np.unique(df["label"])
		labels_map = {label:i for i, label in enumerate(labels)}
		examples = []
		anchors = df["embeddings"].tolist()
		label = df["label"].tolist()
		for i,j in zip(anchors, label):
			examples.append((i,labels_map[j]))
		return examples

	def __len__(self):
		return len(self.examples)
	def __getitem__(self, item):

		if self.mode == "train":
			anchor, neighbour = self.examples[item]
			sample = {"anchor": anchor, "neighbour": neighbour}
		elif self.mode == "val":
			anchor, label = self.examples[item]
			sample = {"anchor": anchor, "label": label}
		return sample
	def collate_fn(self, batch):
		out = torch.tensor([self.embeddings[i["anchor"]] for i in batch]).to(self.device)
		out_2 = torch.tensor([self.embeddings[i["neighbour"]] for i in batch]).to(self.device)
		return {"anchor": out, "neighbour": out_2}

	def collate_fn_val(self, batch):
		out = torch.tensor([i["anchor"] for i in batch]).to(self.device)
		labels = torch.tensor([i["label"] for i in batch]).to(self.device)
		return {"anchor": out, "label": labels}

class SCAN_model(torch.nn.Module):
	def __init__(self, num_labels, dropout, hidden_dim=768):
		super(SCAN_model, self).__init__()
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

	def get_predictions(self, dataset):
		predictions, probs, targets = [], [], []
		epoch_iterator = tqdm(dataset)
		with torch.no_grad():
			for i, batch in enumerate(epoch_iterator):
				self.classifier.eval()
				output_i = self.forward(batch["anchor"])
				probs.extend(torch.nn.functional.softmax(output_i, dim=1).cpu().numpy())
				predictions.extend(torch.argmax(output_i, dim=1).cpu().numpy())
				try:
					targets.extend(batch["label"].cpu().numpy())
				except:
					pass
		out = {'predictions': predictions, 'probabilities': probs, 'targets': targets}
		return out
		
def evaluate(model, val_dataloader):
	with torch.no_grad():
		out = model.get_predictions(val_dataloader)
	hungarian_match_metrics = hungarian_evaluate(np.array(out["targets"]), np.array(out["predictions"]))
	cm = hungarian_match_metrics["confusion matrix"]
	clf_report = hungarian_match_metrics["classification_report"]
	print (fn_val, hungarian_match_metrics)
	del hungarian_match_metrics["classification_report"]

	del hungarian_match_metrics["confusion matrix"]
	print (cm, "\n", clf_report)
	print (fn_val, hungarian_match_metrics)
	print ("ACCURACY", np.round(hungarian_match_metrics["ACC"], 3))
	return hungarian_match_metrics
	

def predict(model, val_dataloader, path):
	with torch.no_grad():
		out = model.get_predictions(val_dataloader)
	#
	hungarian_match_metrics = hungarian_evaluate(np.array(out["targets"]), np.array(out["predictions"]))
	if os.path.exists(os.path.join(args.path, "test_embedded.pkl")):
		fn_val = os.path.join(args.path, "test_embedded.pkl")
	else:
		fn_val = os.path.join(args.path, "train_embedded.pkl")
	df = pd.read_pickle(fn_val)
	labels=np.unique(df["label"])
	if len(labels) > 1:
		if isinstance(labels[0], np.int64):
			label2id = {int(label):i for i, label in enumerate(labels)}
			id2label = {i:j for j,i in label2id.items()}
		else:
			label2id = {label:i for i, label in enumerate(labels)}
			id2label = {i:j for j,i in label2id.items()}


		with open(os.path.join(path, "predictions.txt"), "w") as outfile:
			for i in hungarian_match_metrics["reordered_preds"]:
				outfile.write(str(id2label[i]) + "\n")
		with open(os.path.join(path, "probabilities.txt"), "w") as outfile:
			for i in out["probabilities"]:
				outfile.write("\t".join(list(map(str, i))) + "\n")
		with open(os.path.join(path, "id2label.json"), "w") as outfile:
			json.dump(id2label, outfile)
		with open(os.path.join(path, "label2id.json"), "w") as outfile:
			json.dump(label2id, outfile)


	else:
		with open(os.path.join(path, "predictions.txt"), "w") as outfile:
			for i in out["predictions"]:
				outfile.write(str(i) + "\n")
		with open(os.path.join(path, "probabilities.txt"), "w") as outfile:
			for i in out["probabilities"]:
				outfile.write("\t".join(list(map(str, i))) + "\n")
				

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", type=str, default="20newsgroup", help="")
	parser.add_argument("--predict_filename", type=str, default=None, help="")
	parser.add_argument("--train_file", type=str, default=None, help="")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
		        help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--batch_size", default=64, type=int,
		        help="Batch size per GPU/CPU for training.")
	parser.add_argument("--num_epochs", default=3, type=int,
		        help="Total number of training epochs to perform.")
	parser.add_argument("--num_runs", default=3, type=int,
		        help="Total number of training epochs to perform.")

	parser.add_argument("--learning_rate", default=0.001, type=float,
		        help="The initial learning rate for Adam.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float,
		        help="Epsilon for Adam optimizer.")
	parser.add_argument("--entropy_weight", default=2, type=float,
		        help="Batch size per GPU/CPU for training.")
	parser.add_argument("--entropy_term", type=str, default="entropy", help="")

	parser.add_argument("--dropout", default=0.1, type=float,
		        help="Batch size per GPU/CPU for training.")
	parser.add_argument("--weight_decay", default=0.0, type=float,
		        help="Weight decay if we apply some.")
	parser.add_argument("--num_clusters", default=1, type=int,
		        help="number of clusters, if not defined, we set it to the number of classes in the training set")


	parser.add_argument("--min_clusters", default=10, type=int,
		        help="number of clusters, if not defined, we set it to the number of classes in the training set")
	parser.add_argument("--max_clusters", default=20, type=int,
		        help="number of clusters, if not defined, we set it to the number of classes in the training set")
	parser.add_argument("--stepsize", default=1, type=int,
		        help="number of clusters, if not defined, we set it to the number of classes in the training set")
	

	args = parser.parse_args()

	if args.predict_filename is not None:
		fn_val = os.path.join(args.path, args.predict_filename)
	elif os.path.exists(os.path.join(args.path, "test_embedded.pkl")):
		fn_val = os.path.join(args.path, "test_embedded.pkl")
	else:
		fn_val = os.path.join(args.path, "train_embedded.pkl")

	if args.train_file is not None:
		#fn_train = os.path.join(args.path, os.path.join(args.train_file))
		fn_train = args.train_file
	else:
		fn_train = os.path.join(args.path, "train_neighbours_embeddings.pkl")

	if args.num_clusters == 1:
		df = pd.read_pickle(fn_train)
		num_classes = len(np.unique(df["label"]))
	else:
		num_classes = args.num_clusters

	device = "cuda" if torch.cuda.is_available() else "cpu"
	#device = "cpu"

	# CUDNN
	#torch.backends.cudnn.benchmark = True

	embeddings_df = os.path.join(args.path, "train_embedded.pkl")
	df_embeddings = pd.read_pickle(embeddings_df)	
	train_embeddings = np.array(df_embeddings["embeddings"].tolist())

	train_dataset = ScanDataset(fn_train,train_embeddings, mode="train")
	val_dataset = ScanDataset(fn_val, mode="val")


	results = []
	distortion_scores = []
	for num_classes in range(args.min_clusters, args.max_clusters, args.stepsize):
		model = SCAN_model(num_classes, args.dropout).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
		# Loss function
		criterion = SCANLoss(entropy_weight=args.entropy_weight, entropy=args.entropy_term, experiment=args.path)
		criterion.to(device)
		model.zero_grad()
		model.train()
		train_iterator = range(int(args.num_epochs))

		args.batch_size = max(64, num_classes * 4) # well, if we have 300 clusters, we'd probably want a batchsize bigger than that!

		train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn = train_dataset.collate_fn, batch_size=args.batch_size)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, collate_fn = val_dataset.collate_fn_val, batch_size=args.batch_size)
		for epoch in train_iterator:
			bar_desc = "Epoch %d of %d | Iteration" % (epoch + 1, len(train_iterator))
			epoch_iterator = tqdm(train_dataloader, desc=bar_desc)
			for step, batch in enumerate(epoch_iterator):
				optimizer.zero_grad()
				model.zero_grad()

				anchor, neighbour = batch["anchor"], batch["neighbour"]
				anchors_output, neighbors_output = model(anchor), model(neighbour)
				total_loss, consistency_loss, entropy_loss = criterion(anchors_output, neighbors_output)
				total_loss.backward()
				optimizer.step()
		# get distortion scores for predictions

		predictions = np.array(model.get_predictions(val_dataloader)["predictions"])

		np.save(os.path.join(args.path, "predictions_" + str(num_classes)), predictions)
		#predictions = np.load(os.path.join(args.path, "predictions_" + str(num_classes) + ".npy"))
		distortion = 0
		for cluster in range(num_classes):
			mask = predictions == cluster
			instances = train_embeddings[mask]
			center = instances.mean(axis=0)
			center = np.array([center])

			distances = pairwise_distances(instances, center, metric="euclidean")

			distances = distances ** 2

			# Add the sum of square distance to the distortion
			distortion += distances.sum()
		distortion_scores.append(distortion)


	locator_kwargs = {
	    "curve_nature": "convex",
	    "curve_direction": "decreasing",
	}

	range_n_clusters = list(range(args.min_clusters, args.max_clusters, args.stepsize))
	elbow_locator = KneeLocator(range_n_clusters, distortion_scores, **locator_kwargs)
	print (elbow_locator)

	elbow_value_ = elbow_locator.knee
	print (elbow_value_)
	elbow_score_ = range_n_clusters[range_n_clusters.index(elbow_value_)]
	elbow_label = "elbow at $k={}$".format(elbow_value_)

	plt.plot(range_n_clusters, distortion_scores)

	plt.axvline(elbow_value_, c="tab:red", linestyle="--", label=elbow_label)
	plt.legend()
	plt.savefig(os.path.join(args.path, "optimal_number_of_clusters.png"))
	#plt.show()

	# well, run last time already, because we have the optimal number of clusters!

	num_classes = int(elbow_value_)

	model = SCAN_model(num_classes, args.dropout)
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
	# Loss function
	criterion = SCANLoss(entropy_weight=args.entropy_weight, entropy=args.entropy_term, experiment=args.path)
	criterion.to(device)
	model.zero_grad()
	model.train()
	train_iterator = range(int(args.num_epochs))

	args.batch_size = max(64, num_classes * 4) # well, if we have 300 clusters, we'd probably want a batchsize bigger than that!

	train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn = train_dataset.collate_fn, batch_size=args.batch_size)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, collate_fn = val_dataset.collate_fn_val, batch_size=args.batch_size)
	for epoch in train_iterator:
		bar_desc = "Epoch %d of %d | Iteration" % (epoch + 1, len(train_iterator))
		epoch_iterator = tqdm(train_dataloader, desc=bar_desc)
		for step, batch in enumerate(epoch_iterator):
			optimizer.zero_grad()
			model.zero_grad()

			anchor, neighbour = batch["anchor"], batch["neighbour"]
			anchors_output, neighbors_output = model(anchor), model(neighbour)
			total_loss, consistency_loss, entropy_loss = criterion(anchors_output, neighbors_output)
			total_loss.backward()
			optimizer.step()


	predict(model, val_dataloader, args.path)

	# merge
	df = pd.read_pickle(fn_val)
	out = model.get_predictions(val_dataloader)
	predictions = out["predictions"]
	df["clusters"] = out["predictions"]
	df["probabilities"] = out["probabilities"]
	del df['embeddings']
	df.to_csv(os.path.join(args.path, "docscan_clusters.csv"))
	# right, 
	# bsub -n 1 -R "rusage[mem=12800,ngpus_excl_p=1]" python src/scan_distortion_scores.py --path trump_archive_experimental/ --min_clusters 10 --max_clusters 60 --stepsize 10 
	# bsub -n 1 -R "rusage[mem=12800]" python src/scan_distortion_scores.py --path trump_archive_experimental/ --min_clusters 20 --max_clusters 90 --stepsize 10 



	# bsub -n 1 -R "rusage[mem=12800]" python src/scan_distortion_scores.py --path greenwashing/ --min_clusters 20 --max_clusters 150 --stepsize 10 
	# # bsub -n 1 -R "rusage[mem=25600]" python src/scan_distortion_scores.py --path tweets --min_clusters 25 --max_clusters 325 --stepsize 25

