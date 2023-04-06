from DataAugmentation import DataAugmentation
import torch
from NLPScanModels import DocScanDataset, DocSCAN_Trainer
import numpy as np


class FinetuningThroughSelflabeling:

    def __init__(self, device: str, model_trainer: DocSCAN_Trainer, train_data, train_embeddings, neighbor_dataset, batch_size):
        self.device = device
        self.model_trainer = model_trainer
        self.train_data = train_data
        self.batch_size = batch_size
        self.train_embeddings = train_embeddings
        self.neighbor_dataset = neighbor_dataset

    def mine_prototypes(self, predict_dataset: DocScanDataset, ):

        predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
                                                               collate_fn=predict_dataset.collate_fn_predict,
                                                               batch_size=self.batch_size)

        predictions_train, probabilities_train = self.model_trainer.get_predictions(predict_dataloader)
        targets_map_train = {i: j for j, i in enumerate(np.unique(df_train["label"]))}
        targets_train = [targets_map_train[i] for i in df_train["label"]]
        print(len(targets_train), len(predictions_train))
        evaluate(np.array(targets_train), np.array(predictions_train), mode='train')

        docscan_clusters_train = evaluate(np.array(targets_train), np.array(predictions_train), mode='train')[
            "reordered_preds"]
        df_train["label"] = targets_train
        df_train["clusters"] = docscan_clusters_train
        df_train["probabilities"] = probabilities_train

    def fine_tune_through_selflabeling(self, df_train, model, augmentation):

        # train data
        predict_dataset_train = DocScanDataset(self.neighbor_dataset, self.X, mode="predict",
                                               test_embeddings=self.X, device=self.device)
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

        df_augmented = self.augment(df_train, method=augmentation)

        embeddings_augmented = self.embedd_sentences_method(df_augmented['sentence'], method='SBert_dropout')
        embeddings_augmented = torch.from_numpy(embeddings_augmented)
        # augmented data
        predict_dataset_augmented = DocScanDataset(self.neighbor_dataset, embeddings_augmented, mode="predict",
                                                   test_embeddings=embeddings_augmented, device=self.device)

        predict_dataloader_augmented = torch.utils.data.DataLoader(predict_dataset_augmented, shuffle=False,
                                                                   collate_fn=predict_dataset_augmented.collate_fn_predict,
                                                                   batch_size=self.args.batch_size)

        predictions_augmented, probabilities_augmented = self.get_predictions(model, predict_dataloader_augmented)
        targets_map_augmented = {i: j for j, i in enumerate(np.unique(df_augmented["label"]))}
        targets_augmented = [targets_map_augmented[i] for i in df_augmented["label"]]
        print(len(targets_augmented), len(predictions_augmented))
        evaluate(np.array(targets_augmented), np.array(predictions_augmented), mode='augmented')

        docscan_clusters_augmented = \
        evaluate(np.array(targets_augmented), np.array(predictions_augmented), mode='augmented')["reordered_preds"]
        df_augmented["label"] = targets_augmented
        df_augmented["clusters"] = docscan_clusters_augmented
        df_augmented["probabilities"] = probabilities_augmented

        optimizer = torch.optim.Adam(model.parameters())
        criterion = ConfidenceBasedCE(threshold=0.99, apply_class_balancing=True)
        criterion.to(self.device)

        batch_size = self.args.batch_size

        dataset = list(zip(self.X, embeddings_augmented))
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=self.args.batch_size)

        train_iterator = range(int(self.args.num_epochs))

        targets_map_augmented = {i: j for j, i in enumerate(np.unique(df_augmented["label"]))}
        targets_augmented = [targets_map_augmented[i] for i in df_augmented["label"]]

        for epoch in train_iterator:
            bar_desc = "Epoch %d of %d | num classes %d | Iteration" % (
            epoch + 1, len(train_iterator), self.args.num_classes)
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
        evaluate(np.array(targets_train), np.array(predictions), verbose=0)
        optimizer.zero_grad()
        model.zero_grad()
        return model