import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):

    def __init__(self, data_path, return_index=False):

        self.data_path = data_path
        self.return_index = return_index
        if str(data_path).endswith(".npy"):
            self.embeddings = np.load(data_path)
        elif str(data_path).endswith(".parquet"):
            self.embeddings = pd.read_parquet(data_path)["embedding"].values
            self.embeddings = np.stack(self.embeddings, axis=0)
        else:
            raise ValueError(f"Unsupported data format for EmbDataset: {data_path}")

        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        if self.return_index:
            if np.isscalar(index):
                return tensor_emb, int(index)
            raise ValueError("EmbDataset(return_index=True) expects scalar index.")
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
