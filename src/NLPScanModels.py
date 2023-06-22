import pandas as pd
from tqdm import tqdm
import torch
from transformers import BertModel, BertTokenizer
from utils.losses import SCANLoss, ConfidenceBasedCE
import numpy as np
from torch import nn
from torch.optim import Adam
import random




class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5, number_classes = 20):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, number_classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer



class Dataset_Bert(torch.utils.data.Dataset):

    def __init__(self, df):

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.labels = [label for label in df['cluster']]
        self.texts = [tokenizer(text,
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class DocScanDataset_Bert(torch.utils.data.Dataset):

    def __init__(self, neighbor_df, texts, test_embeddings="", mode="train", device = 'cpu', method = 'SCANLoss'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.neighbor_df = neighbor_df
        self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for text in texts]
        self.mode = mode
        self.device = device
        self.method = method
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
        #lol
        anchors = torch.tensor([i["anchor"] for i in batch]).to(torch.int)
        out = [self.texts[anchor] for anchor in anchors]
        if self.method == 'SCANLoss':
            neighbors = torch.tensor([i["neighbor"] for i in batch]).to(torch.int)
        elif self.method == 'EntropyLoss':
            neighbors = torch.tensor([i["anchor"] for i in batch]).to(torch.int)
        out_2 = [self.texts[neighbor] for neighbor in neighbors]
        return {"anchor": out, "neighbor": out_2}

    def collate_fn_predict(self, batch):
        out = torch.vstack([i["anchor"] for i in batch]).to(self.device)
        return {"anchor": out}



class Bert_Trainer:
    def __init__(self, num_classes: int, device: str):

        self.model = BertClassifier(number_classes=num_classes).to(device)
        self.device = device

    def finetune_BERT_crossentropy(self, train_data,  learning_rate, epochs, batch_size):

        train = Dataset_Bert(train_data)#, Dataset_Bert(val_data)

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        #val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=learning_rate)


        self.model.to(self.device)
        criterion = criterion.to(self.device)

        for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(self.device)
                mask = train_input['attention_mask'].to(self.device)
                input_id = train_input['input_ids'].squeeze(1).to(self.device)

                output = self.model(input_id, mask)

                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                self.model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                    | Train Accuracy: {total_acc_train / len(train_data): .3f}' )

    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def finetune_BERT_SemanticClustering(self, neighbors, texts, batch_size, learning_rate, epochs, method):

        train = DocScanDataset_Bert(neighbors, texts, self.device, method = method)

        train_dataloader = torch.utils.data.DataLoader(train, shuffle=True,
                                                       collate_fn=train.collate_fn,
                                                       batch_size=batch_size)
        # val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

        criterion = SCANLoss()
        criterion.to(self.device)
        optimizer = Adam(self.model.parameters(), lr=learning_rate)

        self.model.to(self.device)
        # criterion = criterion.cuda()

        for epoch in range(epochs):
            bar_desc = "Epoch %d of %d | Iteration" % (epoch + 1, len(train_dataloader))
            epoch_iterator = tqdm(train_dataloader, desc=bar_desc)
            entropy_loss_train = 0
            consistency_loss_train = 0
            total_loss_train = 0
            for batch in epoch_iterator:
                # batch = batch.to(device)
                anchor, neighbor = batch["anchor"], batch["neighbor"]

                mask = anchor[0]['attention_mask'].to(self.device)
                input_id_anchor = anchor[0]['input_ids'].squeeze(1).to(self.device)
                input_id_neighbor = neighbor[0]['input_ids'].squeeze(1).to(self.device)
                for anchor, neighbor in zip(anchor[1:], neighbor[1:]):
                    mask = torch.cat((mask, anchor['attention_mask'].to(self.device)))
                    input_id_anchor = torch.cat((input_id_anchor, anchor['input_ids'].squeeze(1).to(self.device)))
                    input_id_neighbor = torch.cat((input_id_neighbor, neighbor['input_ids'].squeeze(1).to(self.device)))

                anchors_output, neighbors_output = self.model(input_id_anchor, mask), self.model(input_id_neighbor, mask)

                total_loss, consistency_loss, entropy_loss = criterion(anchors_output, neighbors_output)

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.model.zero_grad()

                entropy_loss_train += entropy_loss
                consistency_loss_train += consistency_loss
                total_loss_train += total_loss
            # predictions, probabilities = self.get_predictions(model, predict_dataloader)
            # evaluate(np.array(targets), np.array(predictions),verbose=0)

            print(
                f'Epochs: {epoch + 1} | Train Loss: {total_loss_train / len(texts): .3f} \
                    | Consistency Loss: {consistency_loss_train / len(texts): .3f}  \
            | Entropy Loss: {entropy_loss_train / len(texts): .3f}'
            )
        optimizer.zero_grad()
        self.model.zero_grad()

    def get_predictions(self, test_data: pd.DataFrame):
        test = Dataset_Bert(test_data)

        test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

        self.model.to(self.device)

        predictions_test = []
        probabilities_test = []

        with torch.no_grad():

            for test_input, test_label in test_dataloader:

                mask = test_input['attention_mask'].to(self.device)
                input_id = test_input['input_ids'].squeeze(1).to(self.device)

                output = self.model(input_id, mask)


                output_list = output.tolist()
                predictions = output.argmax(dim=1).tolist()
                probabilities_test.append(output_list[0])
                probabilities_test.append(output_list[1])
                predictions_test.append(predictions[0])
                predictions_test.append(predictions[1])

        return predictions_test, probabilities_test


class DocScanModel(torch.nn.Module):
    def __init__(self, num_labels, dropout, hidden_dim=768, device = 'cpu', secondlayer = False):
        super(DocScanModel, self).__init__()
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(hidden_dim, num_labels)
        if secondlayer:
            self.classifier2 = torch.nn.Linear(hidden_dim * 2, num_labels)
        self.device = device
        #self.device = "cpu"
        self.dropout = dropout
        self.secondlayer = secondlayer

    def forward(self, feature):
        if self.dropout is not None:
            dropout = torch.nn.Dropout(p=self.dropout)
            feature = dropout(feature)
        output = self.classifier(feature)
        if self.secondlayer:
            if self.dropout is not None:
                dropout = torch.nn.Dropout(p=self.dropout)
                hidden_output = dropout(output)
            output = self.classifier2(hidden_output)
        return output

class DocScanDataset(torch.utils.data.Dataset):
    def __init__(self, neighbor_df, embeddings, test_embeddings="", mode="train", device = 'cpu', method = 'SCANLoss' ):
        self.neighbor_df = neighbor_df
        self.embeddings = embeddings
        self.mode = mode
        self.device = device
        self.method = method
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
        '''
        WTF?????
        '''
        anchors = torch.tensor([i["anchor"] for i in batch])
        out = self.embeddings[anchors].to(self.device)
        if self.method == 'SCANLoss':
            neighbors = torch.tensor([i["neighbor"] for i in batch])
        elif self.method == 'EntropyLoss':
            neighbors = torch.tensor([i["anchor"] for i in batch])
        out_2 = self.embeddings[neighbors].to(self.device)
        return {"anchor": out, "neighbor": out_2}

    def collate_fn_predict(self, batch):
        out = torch.vstack([i["anchor"] for i in batch]).to(self.device)
        return {"anchor": out}


class DocScanDataset_old(torch.utils.data.Dataset):
    def __init__(self, neighbor_df, embeddings, test_embeddings="", mode="train", device = 'cpu' ):
        self.neighbor_df = neighbor_df
        self.embeddings = embeddings
        self.mode = mode
        self.device = device
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
        '''
        WTF?????
        '''
        anchors = torch.tensor([i["anchor"] for i in batch])
        out = self.embeddings[anchors].to(self.device)
        neighbors = torch.tensor([i["anchor"] for i in batch])
        out_2 = self.embeddings[neighbors].to(self.device)
        return {"anchor": out, "neighbor": out_2}

    def collate_fn_predict(self, batch):
        out = torch.vstack([i["anchor"] for i in batch]).to(self.device)
        return {"anchor": out}


class DocSCAN_Trainer:
    def __init__(self, num_classes, device, dropout, batch_size, hidden_dim, method):

        self.model = DocScanModel(num_labels=num_classes, dropout=dropout, hidden_dim=hidden_dim, device = device).to(device)
        self.device = device
        self.num_classes = num_classes
        self.dropout = dropout
        self.batch_size = batch_size
        self.method = method

    def get_predictions(self, dataloader):
        predictions, probs = [], []
        epoch_iterator = tqdm(dataloader, total=len(dataloader))
        self.model.eval()
        print(len(dataloader))
        with torch.no_grad():
            for i, batch in enumerate(epoch_iterator):
                self.model.eval()
                output_i = self.model(batch["anchor"])
                # probs.extend(torch.nn.functional.softmax(output_i, dim=1).cpu().numpy())
                probs.extend(output_i.cpu().numpy())
                predictions.extend(torch.argmax(output_i, dim=1).cpu().numpy())
        print(len(predictions))
        return predictions, probs

    def train(self, optimizer, criterion, train_dataloader, num_epochs):
        train_iterator = range(int(num_epochs))
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        softmax = torch.nn.Softmax()
        # train

        #targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
        #targets = [targets_map[i] for i in self.df_test["label"]]

        for epoch in train_iterator:
            bar_desc = "Epoch %d of %d | num classes %d | Iteration" % (epoch + 1, len(train_iterator), self.num_classes)
            epoch_iterator = tqdm(train_dataloader, desc=bar_desc)
            for step, batch in enumerate(epoch_iterator):
                anchor, neighbor = batch["anchor"], batch["neighbor"]

                anchors_output, neighbors_output = self.model(anchor), self.model(neighbor)

                total_loss, consistency_loss, entropy_loss = criterion(anchors_output, neighbors_output)
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.model.zero_grad()
        # predictions, probabilities = self.get_predictions(model, predict_dataloader)
        # evaluate(np.array(targets), np.array(predictions),verbose=0)

        optimizer.zero_grad()
        self.model.zero_grad()


    def train_model(self, neighbor_dataset, train_dataset_embeddings, num_epochs, entropy_weight = 2.0):
        train_dataset = DocScanDataset(neighbor_dataset, train_dataset_embeddings, mode="train", device = self.device, method = self.method)
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = SCANLoss(entropy_weight = entropy_weight)
        criterion.to(self.device)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=train_dataset.collate_fn,
                                                       batch_size=self.batch_size)
        # train
        self.model.train()
        self.train(optimizer, criterion, train_dataloader, num_epochs)

    def give_model(self):

        return self.model

    def train_selflabeling(self, prototype_embeddings, augmented_prototype_embeddings, threshold = 0.99, num_epochs = 5 ):

        self.model.to(self.device)
        self.model.train() #added that
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = ConfidenceBasedCE(device = self.device,threshold=threshold, apply_class_balancing=True)


        dataset = list(zip(prototype_embeddings, augmented_prototype_embeddings))

        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

        train_iterator = range(int(num_epochs))

        for epoch in train_iterator:
            bar_desc = "Epoch %d of %d | num classes %d | Iteration" % (
                epoch + 1, len(train_iterator), self.num_classes)

            epoch_iterator = tqdm(dataloader, desc=bar_desc)
            for step, batch in enumerate(epoch_iterator):
                try:
                    anchor_weak, anchor_strong = batch[0].to(self.device), batch[1].to(self.device)
                    original_output, augmented_output = self.model(anchor_weak), self.model(anchor_strong)
                    total_loss = criterion(original_output, augmented_output)
                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    self.model.zero_grad()
                except ValueError:
                    print(f'Recieved Value Error in step {step}')
        optimizer.zero_grad()
        self.model.zero_grad()















