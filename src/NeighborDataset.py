import pandas as pd
import os
from tqdm import tqdm
from utils.memory import MemoryBank

class Neighbor_Dataset:

    def __init__(self, args,num_neighbors: int, num_classes: int, device: str, path: str, embedding_method: str):

        self.num_neighbors = num_neighbors
        self.num_classes = num_classes
        self.device = device
        self.path = path
        self.embedding_method = embedding_method
        self.args = args


    def _create_neighbor_dataset(self,  memory_bank = None, indices=None,safeNeighborDataset = False):
        if indices is None:
            indices = memory_bank.mine_nearest_neighbors(self.num_neighbors, show_eval=False,
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
        if safeNeighborDataset:
            if self.num_neighbors == 5 and self.args.embedding_model != 'IndicativeSentence':
                df.to_csv(os.path.join(self.path, f"neighbor_dataset_{self.embedding_method}.csv"))
            elif self.num_neighbors == 5 and self.args.embedding_model == 'IndicativeSentence':
                df.to_csv(os.path.join(self.path, f"neighbor_dataset_{self.embedding_method}_indicativesentence_{self.args.indicative_sentence.replace('<','^').replace('>','?').replace(' ','_').replace('!', '5').replace('.', '6')}.csv"))
            else:
                df.to_csv(os.path.join(self.path, f"neighbor_dataset_{self.embedding_method}" + str(self.num_neighbors) + ".csv"))
        return df

    def _retrieve_neighbours_gpu(self, X, batchsize=16384, num_neighbors=5):
        #This is ok since it is not an exact algorithm :)
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

    def create_neighbor_dataset(self, data, createNewDataset = False, safeNeighborDataset = False):

        if os.path.exists(os.path.join(self.path, f"neighbor_dataset_{self.embedding_method}.csv")) and self.num_neighbors == 5 and not createNewDataset and self.args.embedding_model != 'IndicativeSentence':
            print("loading SBERT neighbor dataset")
            neighbor_dataset = pd.read_csv(os.path.join(self.path, f"neighbor_dataset_{self.embedding_method}.csv"))
        elif os.path.exists(os.path.join(self.path, f"neighbor_dataset_{self.embedding_method}_indicativesentence_{self.args.indicative_sentence.replace('<','^').replace('>','?').replace(' ','_').replace('!', '5').replace('.', '6')}.csv")) and self.num_neighbors == 5 and not createNewDataset and self.args.embedding_model == 'IndicativeSentence':
            print("loading IS neighbor dataset")
            neighbor_dataset = pd.read_csv(os.path.join(self.path, f"neighbor_dataset_{self.embedding_method}_indicativesentence_{self.args.indicative_sentence.replace('<','^').replace('>','?').replace(' ','_').replace('!', '5').replace('.', '6')}.csv"))

        elif os.path.exists(os.path.join(self.path, f"neighbor_dataset_{self.embedding_method}" + str(self.num_neighbors) + ".csv")) and not createNewDataset:
            neighbor_dataset = pd.read_csv(
                os.path.join(self.path, f"neighbor_dataset_{self.embedding_method}" + str(self.num_neighbors) + ".csv"))
        else:
            if self.device == "cpu":
                memory_bank = MemoryBank(data, "", len(data),
                                              data.shape[-1],
                                              self.num_classes)
                neighbor_dataset = self._create_neighbor_dataset(memory_bank = memory_bank, safeNeighborDataset = safeNeighborDataset)
            else:
                indices = self._retrieve_neighbours_gpu(data.cpu().numpy(), num_neighbors=self.num_neighbors)
                neighbor_dataset = self._create_neighbor_dataset(indices = indices, safeNeighborDataset = safeNeighborDataset)

        return neighbor_dataset


