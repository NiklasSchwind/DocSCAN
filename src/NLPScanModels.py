from tqdm import tqdm
import torch
from utils.DocSCAN_utils import DocScanDataset, DocScanModel
from utils.losses import SCANLoss
import numpy as np



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



    def finetune_BERT(self,model, train_data,  learning_rate, epochs, device):

        train = Dataset_Bert(train_data)#, Dataset_Bert(val_data)

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
        #val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)



        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)


        model = model.to(device)
        criterion = criterion.to(device)

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
















