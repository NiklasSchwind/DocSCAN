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
from NLPScanModels import DocSCAN_Trainer

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
        embeddings_method = 'IndicativeSentence'
        embedder = Embedder( path = self.args.path, embedding_method = embeddings_method, device = self.args.device)

        self.X = embedder.embed(texts = df_train["sentence"], mode = 'train', createNewEmbeddings= False)
        self.X_test = embedder.embed(texts = self.df_test["sentence"], mode = 'test', createNewEmbeddings = False)

        print("retrieving neighbors...")

        NeighborDataset = Neighbor_Dataset(num_neighbors= self.args.num_neighbors, num_classes = args.num_classes, device = self.args.device, path= self.args.path, embedding_method = embeddings_method)

        self.neighbor_dataset = NeighborDataset.create_neighbor_dataset(self.X, createNewDataset=True)

        targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}

        mode = 'DocSCAN'

        for _ in range(10):

            predict_dataset = DocScanDataset(self.neighbor_dataset, self.X_test, mode="predict",
                                             test_embeddings=self.X_test, device=self.device)
            predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
                                                             collate_fn=predict_dataset.collate_fn_predict,
                                                             batch_size=self.args.batch_size)

            if mode == 'DocSCAN':

                Trainer = DocSCAN_Trainer(num_classes= self.args.num_classes,device = self.device, dropout = self.args.dropout, batch_size= self.args.batch_size, hidden_dim = len(self.X[-1]))
                Trainer.train_model(neighbor_dataset = self.neighbor_dataset, train_dataset_embeddings = self.X, num_epochs = self.args.num_epochs)
                predictions, probabilities = Trainer.get_predictions(predict_dataloader)
                print("docscan trained with n=", self.args.num_classes, "clusters...")

            elif mode == 'PrototypeBert':
                pass





            targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
            targets = [targets_map[i] for i in self.df_test["label"]]
            evaluation.evaluate(np.array(targets), np.array(predictions))
            print(len(targets), len(predictions))
            evaluation.print_statistic_of_latest_experiment()

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