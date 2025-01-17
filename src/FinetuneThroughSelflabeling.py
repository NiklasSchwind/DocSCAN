import pandas as pd
from DataAugmentation import DataAugmentation
import torch
from NLPScanModels import DocScanDataset, DocSCAN_Trainer
import numpy as np
from PrintEvaluation import Evaluation
from Embedder import Embedder
from scipy.special import softmax
import copy
import time
#import nlpaug.augmenter.sentence as nas
import random



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
                 args
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
        self.args = args
        self.num_prototypes = 0



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
        self.num_prototypes = len(df_Prototypes)
        df_Prototypes = df_Prototypes.sample(n=min(self.args.max_prototypes,len(df_Prototypes)))

        return df_Prototypes

    def fine_tune_through_selflabeling(self, augmentation_method = 'bla', giveProtoypes: bool = False):

        # train data
        predict_dataset_train = DocScanDataset(self.neighbor_dataset, self.train_embeddings, mode="predict",
                                               test_embeddings=self.train_embeddings, device=self.device,method = self.clustering_method)

        df_prototypes = self.mine_prototypes(predict_dataset_train)

        df_augmented = copy.deepcopy(df_prototypes)

        #if augmentation_method == 'Backtranslation_fr_en':
        #    df_augmented['sentence'] = self.data_augmenter.backtranslation(list(df_augmented['sentence']), language_order = ['fr','en'])
        #elif augmentation_method == 'Backtranslation_de_en':
        #    df_augmented['sentence'] = self.data_augmenter.backtranslation(list(df_augmented['sentence']), language_order = ['de','en'])

        if augmentation_method == 'Random':
            augmentation_methods = ['Dropout', 'Cropping', 'Deletion', 'Nothing']
            augmentation_method = random.choice(augmentation_methods)

        if augmentation_method == 'Deletion':
            df_augmented['sentence'] = self.data_augmenter.random_deletion(list(df_augmented['sentence']), ratio = self.args.ratio_for_deletion)
            print(df_prototypes)
            print(df_augmented)
        elif augmentation_method == 'Cropping':
            df_augmented['sentence'] = self.data_augmenter.random_cropping(list(df_augmented['sentence']), ratio = self.args.ratio_for_deletion)
        elif augmentation_method == 'Random':
            df_augmented['sentence'] = self.data_augmenter.random_sentence(texts = list(df_augmented['sentence']), alldata = list(self.train_data['sentence']))
        elif augmentation_method == 'Summarization':
            df_augmented['sentence'] = self.data_augmenter.summarize_batch_t5(texts = list(df_augmented['sentence']), t5_model = self.args.t5_model)
        elif augmentation_method == 'Backtranslation':
            start = time.time()

            df_augmented['sentence'] = self.data_augmenter.backtranslation(data=list(df_augmented['sentence']))
            end = time.time()
            print(list(df_prototypes['sentence'])[1:5])
            print('Time: {:5.3f}s'.format(end-start), end='  ')
            print('Hi')
            print(list(df_augmented['sentence'])[1:5])
        elif augmentation_method == 'BacktranslationSentenceLevel':
            df_augmented['sentence'] = self.data_augmenter.sentencelevel_backtranslation(data=list(df_augmented['sentence']))
            print(list(df_prototypes['sentence'])[1:5])
            print('Hi')
            print(list(df_augmented['sentence'])[1:5])
        elif augmentation_method == 'ParaphrasingSentenceLevel':
            df_augmented['sentence'] = self.data_augmenter.sentencelevel_backtranslation(data=list(df_augmented['sentence']))
            print(list(df_prototypes['sentence'])[1:5])
            print('Hi')
            print(list(df_augmented['sentence'])[1:5])
        elif augmentation_method == 'Dropout':
            df_augmented['sentence'] = df_augmented['sentence']
        elif augmentation_method == 'Nothing':
            df_augmented['sentence'] = self.data_augmenter.random_deletion(list(df_augmented['sentence']), ratio = 0)

        #elif augmentation_method == 'LengthIncrease':
        #    aug = nas.ContextualWordEmbsForSentenceAug(model_path='distilgpt2')
        #    df_augmented['sentence'] = aug.augment(list(df_augmented['sentence']))
        elif augmentation_method == 'Paraphrase':
            print(list(df_augmented['sentence'])[1:5])
            df_augmented['sentence'] = self.data_augmenter.paraphrase_texts(list(df_augmented['sentence']), 32, max([len(sentence)+1 for sentence in list(df_augmented['sentence'])]))
            print('Hi')
            print(list(df_augmented['sentence'])[1:5])
        else:
            print('\n\n\nNO DATA AUGMENTATION APPLIED!!!!!!!!!!!!!!\n\n\n')


        embeddings_prototypes = self.embedder.embed(df_prototypes['sentence'], mode = 'embed', createNewEmbeddings = True, safeEmbeddings = False)

        if self.embedder.embedding_method == 'SBert' and augmentation_method == 'Dropout':
            embeddings_augmented = self.data_augmenter.SBert_embed_with_dropout(df_augmented['sentence'], 'sentence-transformers/all-mpnet-base-v2',128)
        elif self.embedder.embedding_method == 'IndicativeSentence' and augmentation_method == 'Dropout':
            embeddings_augmented = self.data_augmenter.embed_IndicativeSentences_with_dropout(texts = df_augmented['sentence'], indicative_sentence=self.embedder.indicative_sentence, indicative_sentence_position=self.embedder.indicative_sentence_position)
        else:
            embeddings_augmented = self.embedder.embed(df_augmented['sentence'], mode='embed', createNewEmbeddings=True,
                                               safeEmbeddings=False)

        self.model_trainer.train_selflabeling(embeddings_prototypes, embeddings_augmented, threshold = self.threshold, num_epochs = 5, augmentation_method = augmentation_method)

        if giveProtoypes:
            return df_prototypes

    def mine_prototype_indexes(self, predict_dataset: DocScanDataset):

        predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
                                                               collate_fn=predict_dataset.collate_fn_predict,
                                                               batch_size=self.batch_size)

        predictions_train, probabilities_train = self.model_trainer.get_predictions(predict_dataloader)
        print(f'list {len(probabilities_train)}')
        targets_map_train = {i: j for j, i in enumerate(np.unique(self.train_data["label"]))}
        targets_train = [targets_map_train[i] for i in self.train_data["label"]]

        docscan_clusters_train = self.evaluator.evaluate(np.array(targets_train), np.array(predictions_train), addToStatistics=False)[
            "reordered_preds"]
        self.train_data["label"] = targets_train
        self.train_data["clusters"] = docscan_clusters_train
        self.train_data["probabilities"] = probabilities_train

        probabilities_array = np.array(self.train_data["probabilities"].tolist())


        softmax_probabilities = np.apply_along_axis(softmax, 1, probabilities_array)


        max_probabilities = np.max(softmax_probabilities, axis=1)


        indices = np.where(max_probabilities >= self.threshold)[0]

        return indices

    def fine_tune_through_selflabeling_fast(self):

        # train data
        predict_dataset_train = DocScanDataset(self.neighbor_dataset, self.train_embeddings, mode="predict",
                                               test_embeddings=self.train_embeddings, device=self.device,method = self.clustering_method)

        prototype_indexes = self.mine_prototype_indexes(predict_dataset_train)
        if len(prototype_indexes) == 0:
            self.num_prototypes = len(prototype_indexes)
        else:
            self.num_prototypes = len(prototype_indexes)
            print(f'Num Prototypes {self.num_prototypes}')
            prototypes = copy.deepcopy(self.train_embeddings[prototype_indexes])
            copy_prototypes  = copy.deepcopy(prototypes)


            self.model_trainer.train_selflabeling(prototypes, copy_prototypes, threshold = self.threshold, num_epochs = 5, augmentation_method = 'Dropout')



    def get_predictions(self, test_data):

        predictions, probabilities = self.model_trainer.get_predictions(test_data)

        return predictions, probabilities




