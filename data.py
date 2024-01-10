import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


token = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir="./model/bert")


class DataSets(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x.iloc[i], self.y.iloc[i]


def collate_fn(data):
    x = [v[0] for v in data]
    y = [v[1] for v in data]

    data = token.batch_encode_plus(batch_text_or_text_pairs=x,
                                   truncation=True,
                                   padding="max_length",
                                   max_length=128,
                                   return_tensors="pt")
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    label = torch.LongTensor(y)
    return input_ids, attention_mask, token_type_ids, label

def load_data(is_train=True):
    data = pd.read_csv("./data/waimai_10k.csv")
    x_train, x_test, y_train, y_test = train_test_split(data.loc[:, "review"], data.loc[:, "label"], test_size=0.3)
    if is_train:
        return DataLoader(
            dataset=DataSets(x_train, y_train),
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )
    else:
        return DataLoader(
            dataset=DataSets(x_test, y_test),
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )


if __name__ == '__main__':
    dataloader = load_data()
    for step, (input_ids, attention_mask, token_type_ids, y) in enumerate(dataloader):
        print(step, input_ids.shape, attention_mask.shape, token_type_ids.shape, y)