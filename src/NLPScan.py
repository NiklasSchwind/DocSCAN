import sys, os, json, argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils.memory import MemoryBank
import torch
from utils.DocSCAN_utils import DocScanDataset, DocScanModel
from utils.losses import SCANLoss
from tqdm import tqdm
from utils.utils import *
from PrintEvaluation import Evaluation
from Embedder import Embedder





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

    def embedd_sentences(self, sentences):
        embedder = SentenceTransformer(self.args.embedding_model)
        embedder.max_seq_length = self.args.max_seq_length
        corpus_embeddings = embedder.encode(sentences, batch_size=32, show_progress_bar=True)
        return corpus_embeddings

    def create_neighbor_dataset(self, indices=None):
        if indices is None:
            indices = self.memory_bank.mine_nearest_neighbors(self.args.num_neighbors, show_eval=False,
                                                              calculate_accuracy=False)
        examples = []
        for i, index in enumerate(indices):
            anchor = i
            neighbors = index
            for neighbor in neighbors:
                if neighbor == i:
                    continue
                examples.append((anchor, neighbor))
        df = pd.DataFrame(examples, columns=["anchor", "neighbor"])
        if self.args.num_neighbors == 5:
            df.to_csv(os.path.join(self.args.path, "neighbor_dataset.csv"))
        else:
            df.to_csv(os.path.join(self.args.path, "neighbor_dataset" + str(self.args.num_neighbors) + ".csv"))
        return df

    def retrieve_neighbours_gpu(self, X, batchsize=16384, num_neighbors=5):
        import faiss
        res = faiss.StandardGpuResources()  # use a single GPU
        n, dim = X.shape[0], X.shape[1]
        index = faiss.IndexFlatIP(dim)  # create CPU index
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)  # create GPU index
        gpu_index_flat.add(X)  # add vectors to the index

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
                # probs.extend(torch.nn.functional.softmax(output_i, dim=1).cpu().numpy())
                probs.extend(output_i.cpu().numpy())
                predictions.extend(torch.argmax(output_i, dim=1).cpu().numpy())
        print(len(predictions))
        return predictions, probs

    def train(self, model, optimizer, criterion, train_dataloader, num_classes):
        train_iterator = range(int(self.args.num_epochs))
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        softmax = torch.nn.Softmax()
        # train

        targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
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
        # predictions, probabilities = self.get_predictions(model, predict_dataloader)
        # evaluate(np.array(targets), np.array(predictions),verbose=0)

        optimizer.zero_grad()
        model.zero_grad()
        return model

    def train_model(self):
        train_dataset = DocScanDataset(self.neighbor_dataset, self.X, mode="train", device = self.device)
        model = DocScanModel(self.args.num_classes, self.args.dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = SCANLoss()
        criterion.to(self.device)

        batch_size = self.args.batch_size
        train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=train_dataset.collate_fn,
                                                       batch_size=batch_size)
        # train
        model = self.train(model, optimizer, criterion, train_dataloader, self.args.num_classes)

        return model

    def run_main(self):
        # embedd using SBERT

        print("loading data...")
        train_data = os.path.join(self.args.path, "train.jsonl")
        test_data = os.path.join(self.args.path, "test.jsonl")

        df_train = self.load_data(train_data)
        args.num_classes = df_train.label.nunique()
        self.df_test = self.load_data(test_data)

        print("embedding sentences...")
        embeddings_method = 'Doc2Vec768'
        embedder_train = Embedder(texts = df_train["sentence"],  path = self.args.path,
                 embedding_method = embeddings_method, device = self.args.device, mode = 'train')
        embedder_test = Embedder(texts = self.df_test["sentence"],  path = self.args.path,
                 embedding_method = embeddings_method, device = self.args.device, mode = 'test')

        self.X = embedder_train.embed(createNewEmbeddings= False)
        self.X_test = embedder_test.embed(createNewEmbeddings = False)

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


        targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}

        for _ in range(10):
            model = self.train_model()
            # test data
            predict_dataset = DocScanDataset(self.neighbor_dataset, self.X_test, mode="predict",
                                             test_embeddings=self.X_test, device= self.device)
            predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
                                                             collate_fn=predict_dataset.collate_fn_predict,
                                                             batch_size=self.args.batch_size)

            predictions, probabilities = self.get_predictions(model, predict_dataloader)


            print("docscan trained with n=", self.args.num_classes, "clusters...")

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