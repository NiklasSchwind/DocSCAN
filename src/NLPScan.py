import sys, os, json, argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from NeighborDataset import Neighbor_Dataset
from utils.memory import MemoryBank
import torch
from utils.DocSCAN_utils import DocScanModel
from utils.losses import SCANLoss
from tqdm import tqdm
from utils.utils import *
from PrintEvaluation import Evaluation
from Embedder import Embedder
from NLPScanModels import DocSCAN_Trainer, Bert_Trainer, DocScanDataset
from scipy.special import softmax
from FinetuneThroughSelflabeling import FinetuningThroughSelflabeling
import random
import numpy as np


seeds = [162562563,36325637,37537389,84876734,674568,474737,37584,48773,15425,7623,5245,52,45252,567889,975432,52542,74557,245241,1341456,7489659,4636551,1341363,7857562,51345]
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
        random.seed(seeds[0])
        np.random.seed(seeds[0])
        torch.manual_seed(seeds[0])
        print("loading data...")
        train_data = os.path.join(self.args.path, "train.jsonl")
        test_data = os.path.join(self.args.path, "test.jsonl")

        df_train = self.load_data(train_data)
        self.args.num_classes = df_train.label.nunique()
        self.df_test = self.load_data(test_data)

        print("embedding sentences...")

        embeddings_method = self.args.embedding_model
        embedder = Embedder( path = self.args.path, embedding_method = embeddings_method, device = self.args.device)

        if self.args.indicative_sentence == 'nothing':
            self.X = embedder.embed(texts = df_train["sentence"], mode = 'train', createNewEmbeddings= self.args.new_embeddings)
            self.X_test = embedder.embed(texts = self.df_test["sentence"], mode = 'test', createNewEmbeddings = self.args.new_embeddings)
        else:
            embedder.set_indicative_sentence(indicative_sentence = self.args.indicative_sentence)
            embedder.set_indicative_sentence_position(indicative_sentence_position = self.args.indicative_sentence_position)
            self.X = embedder.embed(texts=df_train["sentence"], mode='train',
                                    createNewEmbeddings=self.args.new_embeddings)
            self.X_test = embedder.embed(texts=self.df_test["sentence"], mode='test',
                                         createNewEmbeddings=self.args.new_embeddings)
        print("retrieving neighbors...")

        NeighborDataset = Neighbor_Dataset(num_neighbors= self.args.num_neighbors, num_classes = args.num_classes, device = self.args.device, path= self.args.path, embedding_method = embeddings_method, args = self.args)

        self.neighbor_dataset = NeighborDataset.create_neighbor_dataset(self.X, createNewDataset=False)



        targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}

        mode = self.args.model_method    #DocSCAN --> Trains linear classifier on top of embeddings with SCANLoss
                            #PrototypeBert --> Trains linear classifier on top of embeddings with SCANLoss, mines Prototypes in training data and trains BERT classifier with them
                            #DocBERT --> Trains Full Bert Classifier with SCAN loss


        for _ in range(int(self.args.repetitions)):

            random.seed(int(seeds[_]))
            np.random.seed(int(seeds[_]))
            torch.manual_seed(int(seeds[_]))

            if mode == 'DocSCAN':

                predict_dataset = DocScanDataset(self.neighbor_dataset, self.X_test, mode="predict",
                                                 test_embeddings=self.X_test, device=self.device, method = self.args.clustering_method)
                predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
                                                                 collate_fn=predict_dataset.collate_fn_predict,
                                                                 batch_size=self.args.batch_size)

                Trainer = DocSCAN_Trainer(num_classes= self.args.num_classes,device = self.device, dropout = self.args.dropout, batch_size= self.args.batch_size, hidden_dim = len(self.X[-1]), method = self.args.clustering_method)
                Trainer.train_model(neighbor_dataset = self.neighbor_dataset, train_dataset_embeddings = self.X, num_epochs = self.args.num_epochs, entropy_weight=self.args.entropy_weight)
                predictions, probabilities = Trainer.get_predictions(predict_dataloader)
                print("docscan trained with n=", self.args.num_classes, "clusters...")
                targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
                targets = [targets_map[i] for i in self.df_test["label"]]
                evaluation.evaluate(np.array(targets), np.array(predictions))
                print(len(targets), len(predictions))
                evaluation.print_statistic_of_latest_experiment()

            elif mode == 'DocSCAN_finetuning':

                predict_dataset = DocScanDataset(self.neighbor_dataset, self.X_test, mode="predict",
                                                 test_embeddings=self.X_test, device=self.args.device, method = self.args.clustering_method)
                predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
                                                                 collate_fn=predict_dataset.collate_fn_predict,
                                                                 batch_size=self.args.batch_size)

                Trainer = DocSCAN_Trainer(num_classes= self.args.num_classes,device = self.args.device, dropout = self.args.dropout, batch_size= self.args.batch_size, hidden_dim = len(self.X[-1]), method = self.args.clustering_method)
                Trainer.train_model(neighbor_dataset = self.neighbor_dataset, train_dataset_embeddings = self.X, num_epochs = self.args.num_epochs, entropy_weight=self.args.entropy_weight)
                predictions, probabilities = Trainer.get_predictions(predict_dataloader)
                print("docscan trained with n=", self.args.num_classes, "clusters...")
                targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
                targets = [targets_map[i] for i in self.df_test["label"]]
                evaluation_beforeSL.evaluate(np.array(targets), np.array(predictions))
                print(len(targets), len(predictions))
                print('#############Before SelfLabeling: ################################')
                evaluation_beforeSL.print_statistic_of_latest_experiment()

                SelfLabeling = FinetuningThroughSelflabeling(model_trainer=Trainer, evaluation = evaluation_afterSL,
                 embedder = embedder, train_data = df_train, train_embeddings = self.X,
                 neighbor_dataset = self.neighbor_dataset,
                 batch_size = self.args.batch_size, device = self.device, threshold = self.args.threshold, clustering_method = self.args.clustering_method, args = self.args)

                SelfLabeling.fine_tune_through_selflabeling(augmentation_method = self.args.augmentation_method)

                predictions, probabilities = SelfLabeling.get_predictions(predict_dataloader)

                targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
                targets = [targets_map[i] for i in self.df_test["label"]]

                evaluation_afterSL.evaluate(np.array(targets), np.array(predictions))
                evaluation_afterSL.print_statistic_of_latest_experiment()

            elif mode == 'DocSCAN_finetuning_multi':

                accuracy_development = []
                prototype_number_development = [0]
                predict_dataset = DocScanDataset(self.neighbor_dataset, self.X_test, mode="predict",
                                                 test_embeddings=self.X_test, device=self.args.device, method = self.args.clustering_method)
                predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
                                                                 collate_fn=predict_dataset.collate_fn_predict,
                                                                 batch_size=self.args.batch_size)

                Trainer = DocSCAN_Trainer(num_classes= self.args.num_classes,device = self.args.device, dropout = self.args.dropout, batch_size= self.args.batch_size, hidden_dim = len(self.X[-1]), method = self.args.clustering_method)
                Trainer.train_model(neighbor_dataset = self.neighbor_dataset, train_dataset_embeddings = self.X, num_epochs = self.args.num_epochs,  entropy_weight=self.args.entropy_weight)
                predictions, probabilities = Trainer.get_predictions(predict_dataloader)
                print("docscan trained with n=", self.args.num_classes, "clusters...")
                targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
                targets = [targets_map[i] for i in self.df_test["label"]]
                metrics = evaluation_beforeSL.evaluate(np.array(targets), np.array(predictions))
                accuracy_development.append(metrics['full_statistics']["accuracy"])
                print(len(targets), len(predictions))
                print('#############Before SelfLabeling: ################################')
                evaluation_beforeSL.print_statistic_of_latest_experiment()

                SelfLabeling = FinetuningThroughSelflabeling(model_trainer=Trainer, evaluation = evaluation_afterSL,
                 embedder = embedder, train_data = df_train, train_embeddings = self.X,
                 neighbor_dataset = self.neighbor_dataset,
                 batch_size = self.args.batch_size, device = self.device, threshold = self.args.threshold, clustering_method = self.args.clustering_method, args = self.args)

                num_prototypes_before = SelfLabeling.num_prototypes
                num_prototypes = SelfLabeling.num_prototypes + 1

                while num_prototypes_before < num_prototypes:

                    num_prototypes_before = num_prototypes
                    prototypes = SelfLabeling.fine_tune_through_selflabeling(augmentation_method = self.args.augmentation_method, giveProtoypes = True)
                    num_prototypes = SelfLabeling.num_prototypes

                    print(f'Number prototypes: {SelfLabeling.num_prototypes}')

                    predictions, probabilities = SelfLabeling.get_predictions(predict_dataloader)

                    targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
                    targets = [targets_map[i] for i in self.df_test["label"]]

                    metrics = evaluation_afterSL.evaluate(np.array(targets), np.array(predictions), addToStatistics=False, doPrint=True)
                    accuracy_development.append(metrics['full_statistics']["accuracy"])
                    prototype_number_development.append(SelfLabeling.num_prototypes)


                print(accuracy_development)
                print(prototype_number_development)

                evaluation_afterSL.evaluate(np.array(targets), np.array(predictions))
                evaluation_afterSL.print_statistic_of_latest_experiment()


            elif mode == 'PrototypeBert':

                #Train DocSCAN model with train dataset to mine Protoypes
                PrototypeMine_Trainer = DocSCAN_Trainer(num_classes=self.args.num_classes, device=self.device,
                                          dropout=self.args.dropout, batch_size=self.args.batch_size,
                                          hidden_dim=len(self.X[-1]), method = self.args.clustering_method)
                PrototypeMine_Trainer.train_model(neighbor_dataset=self.neighbor_dataset, train_dataset_embeddings=self.X,
                                    num_epochs=self.args.num_epochs,  entropy_weight=self.args.entropy_weight)
                # Predict train dataset to receive class probabilities
                predict_dataset_train = DocScanDataset(self.neighbor_dataset, self.X, mode="predict",
                                                       test_embeddings=self.X, device=self.device, method = self.args.clustering_method)
                predict_dataloader_train = torch.utils.data.DataLoader(predict_dataset_train, shuffle=False,
                                                                       collate_fn=predict_dataset_train.collate_fn_predict,
                                                                       batch_size=self.args.batch_size)

                predictions_train, probabilities_train = PrototypeMine_Trainer.get_predictions(predict_dataloader_train)

                targets_map_train = {i: j for j, i in enumerate(np.unique(df_train["label"]))}
                targets_train = [targets_map_train[i] for i in df_train["label"]]

                docscan_clusters_train = evaluation.evaluate(np.array(targets_train), np.array(predictions_train), addToStatistics=False)

                df_train["label"] = targets_train
                df_train["clusters"] = docscan_clusters_train["reordered_preds"]
                df_train["probabilities"] = probabilities_train
                # Mine prototypes from predictions
                df_ExtraModel = df_train[df_train["probabilities"].apply(softmax).apply(np.max) >= self.args.threshold]
                df_ExtraModel = df_ExtraModel[['sentence', 'clusters']].rename(
                    {'sentence': 'text', 'clusters': 'cluster'}, axis='columns')

                # Train BERT classifier with prototypes
                Extra_Model_Trainer = Bert_Trainer(num_classes=self.args.num_classes, device = self.device )
                Extra_Model_Trainer.finetune_BERT_crossentropy(train_data=df_ExtraModel, learning_rate=1e-5,epochs= 5,batch_size= 32)

                targets_map_test = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
                targets_test = [targets_map_test[i] for i in self.df_test["label"]]

                df_ExtraModel_test = self.df_test
                df_ExtraModel_test['targets'] = targets_test
                df_ExtraModel_test = df_ExtraModel_test[['sentence', 'targets']].rename(
                    {'sentence': 'text', 'targets': 'cluster'},
                    axis='columns')


                predictions_test, probabilities_test = Extra_Model_Trainer.get_predictions(df_ExtraModel_test)

                targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
                targets = [targets_map[i] for i in self.df_test["label"]]

                evaluation.evaluate(targets, predictions_test)
                evaluation.print_statistic_of_latest_experiment()

            elif mode == 'DocBert':

                #Get full Bert classifier with own embeddings
                BERT_trainer = Bert_Trainer(num_classes=self.args.num_classes, device = self.device )

                #Fine Tune full classifier with neighbor dataset and SCAN loss
                BERT_trainer.finetune_BERT_SemanticClustering(self.neighbor_dataset, [text for text in df_train["sentence"]],  self.args.batch_size, 1e-6, self.args.num_epochs, method = self.args.clustering_method)

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
                evaluation.print_statistic_of_latest_experiment()

            elif mode == 'PrototypeAccuracy':

                predict_dataset = DocScanDataset(self.neighbor_dataset, self.X, mode="predict",
                                                 test_embeddings=self.X, device=self.device,
                                                 method=self.args.clustering_method)
                predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
                                                                 collate_fn=predict_dataset.collate_fn_predict,
                                                                 batch_size=self.args.batch_size)

                PrototypeMine_Trainer = DocSCAN_Trainer(num_classes=self.args.num_classes, device=self.device,
                                          dropout=self.args.dropout, batch_size=self.args.batch_size,
                                          hidden_dim=len(self.X[-1]), method = self.args.clustering_method)
                PrototypeMine_Trainer.train_model(neighbor_dataset=self.neighbor_dataset, train_dataset_embeddings=self.X,
                                    num_epochs=self.args.num_epochs, entropy_weight=self.args.entropy_weight)
                predictions, probabilities = PrototypeMine_Trainer.get_predictions(predict_dataloader)
                print("docscan trained with n=", self.args.num_classes, "clusters...")
                targets_map = {i: j for j, i in enumerate(np.unique(df_train["label"]))}
                targets = [targets_map[i] for i in df_train["label"]]
                evaluation_beforeSL.evaluate(np.array(targets), np.array(predictions))
                evaluation_beforeSL.print_statistic_of_latest_experiment()

                predict_dataset_train = DocScanDataset(self.neighbor_dataset, self.X, mode="predict",
                                                       test_embeddings=self.X, device=self.device, method = self.args.clustering_method)
                predict_dataloader_train = torch.utils.data.DataLoader(predict_dataset_train, shuffle=False,
                                                                       collate_fn=predict_dataset_train.collate_fn_predict,
                                                                       batch_size=self.args.batch_size)

                predictions_train, probabilities_train = PrototypeMine_Trainer.get_predictions(predict_dataloader_train)

                targets_map_train = {i: j for j, i in enumerate(np.unique(df_train["label"]))}
                targets_train = [targets_map_train[i] for i in df_train["label"]]

                docscan_clusters_train = evaluation_beforeSL.evaluate(np.array(targets_train), np.array(predictions_train), addToStatistics=False)

                df_train["label"] = targets_train
                df_train["clusters"] = docscan_clusters_train["reordered_preds"]
                df_train["probabilities"] = probabilities_train
                # Mine prototypes from predictions
                df_ExtraModel = df_train[df_train["probabilities"].apply(softmax).apply(np.max) >= self.args.threshold]
                df_ExtraModel = df_ExtraModel[['sentence', 'clusters','label']].rename(
                    {'sentence': 'text', 'clusters': 'cluster'}, axis='columns')
                if len(df_ExtraModel["label"]) != 0:
                    evaluation_afterSL.evaluate(df_ExtraModel["label"], df_ExtraModel["cluster"])
                    evaluation_afterSL.print_statistic_of_latest_experiment()
                else:
                    print(f'############!!!!!!!!!!NO PROTOTYPES found in Experiment {_}!!!!!!!!################')

        if self.args.model_method == 'DocSCAN_finetuning' or self.args.model_method == 'PrototypeAccuracy' or self.args.model_method == 'DocSCAN_finetuning_multi':
            evaluation_beforeSL.print_full_statistics()
            evaluation_afterSL.print_full_statistics()
        else:
            evaluation.print_full_statistics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to output path where output of docscan gets saved")
    parser.add_argument("--embedding_model", default="SBert", type=str,
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
    parser.add_argument("--clustering_method", default='SCANLoss', type=str, help="Choose between SCANLoss and EntropyLoss")
    parser.add_argument("--model_method", default='DocSCAN', type=str,
                        help="Choose between DocSCAN, DocBert and PrototypeBert")
    parser.add_argument("--threshold", default=0.95, type=float,
                        help="threshold for selvlabeling step")
    parser.add_argument("--new_embeddings", default='False', type=str,
                        help="should the embeddings be calculated again")
    parser.add_argument("--augmentation_method", default='Backtranslation_fr_en', type=str,
                        help="can be 'Cropping' or 'Backtranslation_fr_en' for now")
    parser.add_argument("--entropy_weight", default=2.0, type=float,
                        help="adjust the Entropy Weight")
    parser.add_argument("--ratio_for_deletion", default=0.2, type=float,
                        help="adjust the Entropy Weight")
    parser.add_argument("--repetitions", default=3, type=float,
                        help="Number Repetitions of Experiment")
    parser.add_argument("--indicative_sentence", default='nothing', type=str,
                        help="Indicative Sentence to use or nothing")
    parser.add_argument("--indicative_sentence_position", default='first', type=str,
                        help="first or last")
    parser.add_argument("--t5_model", default='large', type=str,
                        help="can be large, base and small")
    parser.add_argument("--show_bars", default='False', type=str,
                        help="True or False")
    parser.add_argument("--max_prototypes", default=5000, type=int,
                        help="maximum number of prototypes")
    args = parser.parse_args()

    if args.dropout == 0:
        args.dropout = None
    if args.outfile != 'NO':
        sys.stdout = open(args.outfile, 'wt')
    args.indicative_sentence  = args.indicative_sentence.replace('^','<').replace('?','>').replace('_',' ').replace('5', '!')

    if args.new_embeddings == 'False':
        args.new_embeddings = False
    elif args.new_embeddings == 'True':
        args.new_embeddings = True

    if args.show_bars == 'False':
        args.show_bars = False
    elif args.show_bars == 'True':
        args.show_bars = True


    docscan = DocSCANPipeline(args)
    if args.model_method == 'DocSCAN_finetuning' or args.model_method == 'PrototypeAccuracy' or args.model_method == 'DocSCAN_finetuning_multi':
        evaluation_beforeSL = Evaluation(name_dataset=args.path, name_embeddings=args.embedding_model)
        evaluation_afterSL = Evaluation(name_dataset=args.path, name_embeddings=args.embedding_model)

    else:
        evaluation = Evaluation(name_dataset = args.path, name_embeddings = args.embedding_model)
    docscan.run_main()