

class neighbor_dataset:
    def __init__(self, num_neighbors: int):

        self.num_neighbors = num_neighbors


    def create_neighbor_dataset(self, indices=None):
        if indices is None:
            indices = self.memory_bank.mine_nearest_neighbors(self.args.num_neighbors, show_eval=False,
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
        if self.args.num_neighbors == 5:
            df.to_csv(os.path.join(self.args.path, "neighbor_dataset.csv"))
        else:
            df.to_csv(os.path.join(self.args.path, "neighbor_dataset" + str(self.args.num_neighbors) + ".csv"))
        return df

    def retrieve_neighbours_gpu(self, X, batchsize=16384, num_neighbors=5):
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

    def



if os.path.exists(os.path.join(self.args.path, "neighbor_dataset.csv")) and self.args.num_neighbors == 5:
    print("loading neighbor dataset")
    self.neighbor_dataset = pd.read_csv(os.path.join(self.args.path, "neighbor_dataset.csv"))
elif os.path.exists(os.path.join(self.args.path, "neighbor_dataset" + str(self.args.num_neighbors) + ".csv")):
    self.neighbor_dataset = pd.read_csv(
        os.path.join(self.args.path, "neighbor_dataset" + str(self.args.num_neighbors) + ".csv"))
else:
    if self.device == "cpu":
        self.memory_bank = MemoryBank(self.X, "", len(self.X),
                                      self.X.shape[-1],
                                      self.args.num_classes)
        self.neighbor_dataset = self.create_neighbor_dataset()
    else:
        indices = self.retrieve_neighbours_gpu(self.X.numpy(), num_neighbors=self.args.num_neighbors)
        self.neighbor_dataset = self.create_neighbor_dataset(indices=indices)