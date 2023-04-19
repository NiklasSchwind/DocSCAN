import pandas as pd
from DataAugmentation import DataAugmentation
import torch
from NLPScanModels import DocScanDataset, DocSCAN_Trainer
import numpy as np
from PrintEvaluation import Evaluation
from Embedder import Embedder
from scipy.special import softmax



class FinetuningThroughSelflabeling:

    def __init__(self,
                 model_trainer: DocSCAN_Trainer,
                 evaluation: Evaluation,
                 embedder: Embedder,
                 train_data: pd.DataFrame,
                 train_embeddings: torch.tensor,
                 neighbor_dataset: pd.DataFrame,
                 batch_size: int,
                 device: str,
                 threshold: float,
                 clustering_method: str,
                 ):
        self.device = device
        self.model_trainer = model_trainer
        self.train_data = train_data
        self.batch_size = batch_size
        self.train_embeddings = train_embeddings
        self.neighbor_dataset = neighbor_dataset
        self.evaluator = evaluation
        self.embedder = embedder
        self.threshold = threshold
        self.clustering_method = clustering_method
        self.data_augmenter = DataAugmentation(device = self.device, batch_size = self.batch_size)

    def mine_prototypes(self, predict_dataset: DocScanDataset):

        predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
                                                               collate_fn=predict_dataset.collate_fn_predict,
                                                               batch_size=self.batch_size)

        predictions_train, probabilities_train = self.model_trainer.get_predictions(predict_dataloader)
        targets_map_train = {i: j for j, i in enumerate(np.unique(self.train_data["label"]))}
        targets_train = [targets_map_train[i] for i in self.train_data["label"]]

        docscan_clusters_train = self.evaluator.evaluate(np.array(targets_train), np.array(predictions_train), addToStatistics=False)[
            "reordered_preds"]
        self.train_data["label"] = targets_train
        self.train_data["clusters"] = docscan_clusters_train
        self.train_data["probabilities"] = probabilities_train

        df_Prototypes = self.train_data[self.train_data["probabilities"].apply(softmax).apply(np.max) >= self.threshold]

        return df_Prototypes

    def fine_tune_through_selflabeling(self, augmentation_method = 'bla'):

        # train data
        predict_dataset_train = DocScanDataset(self.neighbor_dataset, self.train_embeddings, mode="predict",
                                               test_embeddings=self.train_embeddings, device=self.device,method = self.clustering_method)

        df_prototypes = self.mine_prototypes(predict_dataset_train)

        df_augmented = df_prototypes

        if augmentation_method == 'Backtranslation_fr_en':
            df_augmented['sentence'] = self.data_augmenter.backtranslation(df_prototypes['sentence'], language_order = ['fr','en'])
        elif augmentation_method == 'Cropping':
            df_augmented['sentence'] = self.data_augmenter.random_deletion(df_augmented['sentence'], ratio = 0.2)

        embeddings_prototypes = self.embedder.embed(df_augmented['sentence'], mode='embed', createNewEmbeddings=True,safeEmbeddings=False)
        embeddings_augmented = self.embedder.embed(df_augmented['sentence'], mode = 'embed', createNewEmbeddings=True,safeEmbeddings=False)

        self.model_trainer.train_selflabeling(embeddings_prototypes.to(), embeddings_augmented, threshold = self.threshold, num_epochs = 5)


    def get_predictions(self, test_data):

        predictions, probabilities = self.model_trainer.get_predictions(test_data)

        return predictions, probabilities
