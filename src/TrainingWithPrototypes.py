import pandas as pd
from torch import nn
from transformers import BertModel
import torch
import numpy as np
from transformers import BertTokenizer
from torch.optim import Adam
from tqdm import tqdm
from utils.losses import SCANLoss
from utils.DocSCAN_utils import DocScanDataset_BertFinetune

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



tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


class Dataset_Bert(torch.utils.data.Dataset):

    def __init__(self, df):

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


def finetune_BERT(model, train_data,  learning_rate, epochs):
    train = Dataset_Bert(train_data)#, Dataset_Bert(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    #val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0
        '''
        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        '''
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f}' )



#

def finetune_BERT_SemanticClustering(model, neighbors, texts, batch_size,  learning_rate, epochs):
    train = DocScanDataset_BertFinetune(neighbors, [tokenizer(text,padding='max_length', max_length = 512, truncation=True,return_tensors="pt") for text in texts])

    train_dataloader = torch.utils.data.DataLoader(train, shuffle=True,
															 collate_fn=train.collate_fn,
															 batch_size=batch_size)
    #val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = SCANLoss()
    criterion.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        #criterion = criterion.cuda()

    for epoch in range(epochs):
        bar_desc = "Epoch %d of %d | Iteration" % (epoch + 1, len(train_dataloader))
        epoch_iterator = tqdm(train_dataloader, desc=bar_desc)
        entropy_loss_train = 0
        consistency_loss_train = 0
        total_loss_train = 0
        for batch in epoch_iterator:
            #batch = batch.to(device)
            anchor, neighbor = batch["anchor"], batch["neighbor"]
            print(len(anchor))
            print(len(neighbor))
            #for anchor, neighbor in zip(anchor,neighbor):
            mask = anchor['attention_mask'].to(device)
            input_id_anchor = anchor['input_ids'].squeeze(1).to(device)
            input_id_neighbor = neighbor['input_ids'].squeeze(1).to(device)

            anchors_output, neighbors_output = model(input_id_anchor,mask), model(input_id_neighbor,mask)

            total_loss, consistency_loss, entropy_loss = criterion(anchors_output, neighbors_output)

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

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
    model.zero_grad()




def evaluate_Bert(model, test_data):
    test = Dataset_Bert(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

    return total_acc_test / len(test_data)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def get_predictions_Bert(model, test_sentences):



    test = Dataset_Bert(test_sentences)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    predictions_test = []
    probabilities_test = []
    with torch.no_grad():

        for test_input in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            probabilities_test.append(output)
            predictions_test.append(output.argmax(dim=1).item())




    return predictions_test, probabilities_test