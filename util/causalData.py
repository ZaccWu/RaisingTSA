import torch

import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader


class CausalData(Dataset.Dataset):
    def __init__(self, preData, postData, Treat):
        self.preData = preData
        self.postData = postData
        self.Treat = Treat

    def __len__(self):
        return len(self.Treat)

    def __getitem__(self, index):
        predata = torch.Tensor(self.preData[index])
        postdata = torch.Tensor(self.postData[index])
        treat = torch.IntTensor(self.Treat[index])
        return predata, postdata, treat