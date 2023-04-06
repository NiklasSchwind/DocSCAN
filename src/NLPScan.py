import sys, os, json, argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from NeighborDataset import Neighbor_Dataset
from utils.memory import MemoryBank
import torch
from utils.DocSCAN_utils import DocScanDataset, DocScanModel
from utils.losses import SCANLoss
from tqdm import tqdm
from utils.utils import *
from PrintEvaluation import Evaluation
from Embedder import Embedder
from NLPScanModels import DocSCAN_Trainer, Bert_Trainer
from scipy.special import softmax

class DocSCANPipeline():
    def __init__(self, args):
        self.args = args
        self.device = args.device
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

    def run_main(self):

        print("loading data...")
        train_data = os.path.join(self.args.path, "train.jsonl")
        test_data = os.path.join(self.args.path, "test.jsonl")

        df_train = self.load_data(train_data)
        self.args.num_classes = df_train.label.nunique()
        self.df_test = self.load_data(test_data)

        print("embedding sentences...")
        embeddings_method = 'SimCSEsupervised'
        embedder = Embedder( path = self.args.path, embedding_method = embeddings_method, device = self.args.device)

        self.X = embedder.embed(texts = df_train["sentence"], mode = 'train', createNewEmbeddings= True)
        self.X_test = embedder.embed(texts = self.df_test["sentence"], mode = 'test', createNewEmbeddings = True)

        print("retrieving neighbors...")

        NeighborDataset = Neighbor_Dataset(num_neighbors= self.args.num_neighbors, num_classes = args.num_classes, device = self.args.device, path= self.args.path, embedding_method = embeddings_method)

        self.neighbor_dataset = NeighborDataset.create_neighbor_dataset(self.X, createNewDataset=True)

        targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}

        mode = 'DocSCAN'    #DocSCAN --> Trains linear classifier on top of embeddings with SCANLoss
                            #PrototypeBert --> Trains linear classifier on top of embeddings with SCANLoss, mines Prototypes in training data and trains BERT classifier with them
                            #DocBERT --> Trains Full Bert Classifier with SCAN loss


        for _ in range(10):

            if mode == 'DocSCAN':

                predict_dataset = DocScanDataset(self.neighbor_dataset, self.X_test, mode="predict",
                                                 test_embeddings=self.X_test, device=self.device)
                predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
                                                                 collate_fn=predict_dataset.collate_fn_predict,
                                                                 batch_size=self.args.batch_size)

                Trainer = DocSCAN_Trainer(num_classes= self.args.num_classes,device = self.device, dropout = self.args.dropout, batch_size= self.args.batch_size, hidden_dim = len(self.X[-1]))
                Trainer.train_model(neighbor_dataset = self.neighbor_dataset, train_dataset_embeddings = self.X, num_epochs = self.args.num_epochs)
                predictions, probabilities = Trainer.get_predictions(predict_dataloader)
                print("docscan trained with n=", self.args.num_classes, "clusters...")
                targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
                targets = [targets_map[i] for i in self.df_test["label"]]
                evaluation.evaluate(np.array(targets), np.array(predictions))
                print(len(targets), len(predictions))
                evaluation.print_statistic_of_latest_experiment()

            elif mode == 'PrototypeBert':

                #Train DocSCAN model with train dataset to mine Protoypes
                PrototypeMine_Trainer = DocSCAN_Trainer(num_classes=self.args.num_classes, device=self.device,
                                          dropout=self.args.dropout, batch_size=self.args.batch_size,
                                          hidden_dim=len(self.X[-1]))
                PrototypeMine_Trainer.train_model(neighbor_dataset=self.neighbor_dataset, train_dataset_embeddings=self.X,
                                    num_epochs=self.args.num_epochs)
                # Predict train dataset to receive class probabilities
                predict_dataset_train = DocScanDataset(self.neighbor_dataset, self.X, mode="predict",
                                                       test_embeddings=self.X, device=self.device)
                predict_dataloader_train = torch.utils.data.DataLoader(predict_dataset_train, shuffle=False,
                                                                       collate_fn=predict_dataset_train.collate_fn_predict,
                                                                       batch_size=self.args.batch_size)

                predictions_train, probabilities_train = Trainer.get_predictions(predict_dataloader_train)

                targets_map_train = {i: j for j, i in enumerate(np.unique(df_train["label"]))}
                targets_train = [targets_map_train[i] for i in df_train["label"]]

                docscan_clusters_train = evaluation.evaluate(np.array(targets_train), np.array(predictions_train), addToStatistics=False)

                df_train["label"] = targets_train
                df_train["clusters"] = docscan_clusters_train["reordered_preds"]
                df_train["probabilities"] = probabilities_train
                # Mine prototypes from predictions
                df_ExtraModel = df_train[df_train["probabilities"].apply(softmax).apply(np.max) >= 0.95]
                df_ExtraModel = df_ExtraModel[['sentence', 'clusters']].rename(
                    {'sentence': 'text', 'clusters': 'cluster'}, axis='columns')

                # Train BERT classifier with prototypes
                Extra_Model_Trainer = Bert_Trainer(num_classes=self.args.num_classes, device = self.device )
                Extra_Model_Trainer.finetune_BERT_crossentropy(df_ExtraModel, 1e-6, 7, self.device)

                df_ExtraModel_test = self.df_test
                df_ExtraModel_test = df_ExtraModel_test[['sentence', 'label']].rename(
                    {'sentence': 'text', 'label': 'cluster'},
                    axis='columns')

                predictions_test, probabilities_test = Extra_Model_Trainer.get_predictions(df_ExtraModel_test)

                targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
                targets = [targets_map[i] for i in self.df_test["label"]]

                evaluation.evaluate(targets, predictions_test)

            elif mode == 'DocBert':

                #Get full Bert classifier with own embeddings
                BERT_trainer = Bert_Trainer(num_classes=self.args.num_classes, device = self.device )

                #Fine Tune full classifier with neighbor dataset and SCAN loss
                BERT_trainer.finetune_BERT_SemanticClustering(self.neighbor_dataset, [text for text in df_train["sentence"]],  self.args.batch_size, 1e-6, self.args.num_epochs)

                print("docscan trained with n=", self.args.num_classes, "clusters...")

                targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
                targets = [targets_map[i] for i in self.df_test["label"]]

                self.df_test['label'] = targets

                df_ExtraModel_test = self.df_test
                df_ExtraModel_test = df_ExtraModel_test[['sentence', 'label']].rename(
                    {'sentence': 'text', 'label': 'cluster'},
                    axis='columns')

                predictions, probabilities = BERT_trainer.get_predictions(df_ExtraModel_test)

                df_ExtraModel_test = df_ExtraModel_test.rename({'cluster': 'label'},
                                                               axis='columns')

                targets_map = {i: j for j, i in enumerate(np.unique(df_ExtraModel_test["label"]))}
                targets = [targets_map[i] for i in df_ExtraModel_test["label"]]

                print(len(targets), len(predictions))

                evaluation.evaluate(np.array(targets), np.array(predictions))



        evaluation.print_full_statistics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to output path where output of docscan gets saved")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-mpnet-base-v2", type=str,
                        help="SBERT model to use to embedd sentences")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="max seq length of sbert model, sequences longer than this get truncated at this value")
    parser.add_argument("--topk", type=int, default=5, help="numbers of neighbors retrieved to build SCAN training set")
    parser.add_argument("--num_classes", type=int, default=10, help="numbers of clusters")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout for DocSCAN model")
    parser.add_argument("--num_epochs", default=5, type=int, help="number of epochs to train DocSCAN model")
    parser.add_argument("--num_neighbors", default=5, type=int, help="number of epochs to train DocSCAN model")
    parser.add_argument("--device", default='cpu', type=str, help="device the code should be run on")
    parser.add_argument("--outfile", default='NO', type=str, help="file to print outputs to")
    args = parser.parse_args()

    if args.dropout == 0:
        args.dropout = None
    if args.outfile != 'NO':
        sys.stdout = open(args.outfile, 'wt')


    docscan = DocSCANPipeline(args)
    evaluation = Evaluation(name_dataset = args.path, name_embeddings = args.embedding_model)
    docscan.run_main()