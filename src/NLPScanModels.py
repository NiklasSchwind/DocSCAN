from tqdm import tqdm
import torch
from transformers import BertModel, BertTokenizer
from utils.DocSCAN_utils import DocScanDataset, DocScanModel
from utils.losses import SCANLoss
import numpy as np
from torch import nn
from torch.optim import Adam





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



class Trainer_Bert:
    def __init__(self, num_classes: int, ):

        self.model = BertClassifier(number_classes=num_classes)

    def finetune_BERT(self, train_data,  learning_rate, epochs, device):

        train = Dataset_Bert(train_data)#, Dataset_Bert(val_data)

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
        #val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=learning_rate)


        self.model.to(device)
        criterion = criterion.to(device)

        for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

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

    def get_predictions_Bert(self,model, test_data, device):
        test = Dataset_Bert(test_data)

        test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

        # use_cuda = torch.cuda.is_available()
        # device = torch.device("cuda" if use_cuda else "cpu")

        self.model.to(device)

        predictions_test = []
        probabilities_test = []
        i = 0
        with torch.no_grad():

            for test_input, test_label in test_dataloader:

                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                if i <= 10:
                    print(output)
                    print(output.argmax(dim=1))
                    print(output.tolist())
                    print(output.argmax(dim=1).tolist())
                    i += 1
                output_list = output.tolist()
                predictions = output.argmax(dim=1).tolist()
                probabilities_test.append(output_list[0])
                probabilities_test.append(output_list[1])
                predictions_test.append(predictions[0])
                predictions_test.append(predictions[1])

        return predictions_test, probabilities_test















