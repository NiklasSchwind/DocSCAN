from tqdm import tqdm
import torch
from utils.DocSCAN_utils import DocScanDataset, DocScanModel
from utils.losses import SCANLoss
import numpy as np






class NLPScan:
    def __init__(self, ):


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

    def train(self, model, optimizer, criterion, train_dataloader, num_classes, epochs):
        train_iterator = range(int(epochs))
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

    def train_model(self, ):
        train_dataset = DocScanDataset(self.neighbor_dataset, self.X, mode="train", device=self.device)
        model = DocScanModel(self.args.num_classes, self.args.dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = SCANLoss()
        criterion.to(self.device)

        batch_size = self.args.batch_size
        train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                                       collate_fn=train_dataset.collate_fn,
                                                       batch_size=batch_size)
        # train
        model = self.train(model, optimizer, criterion, train_dataloader, self.args.num_classes)

        return model

