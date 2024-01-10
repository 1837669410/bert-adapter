import torch.nn as nn
from transformers import BertModel


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.dense1 = nn.Linear(in_features=768, out_features=64)
        self.dense2 = nn.Linear(in_features=64, out_features=768)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.gelu(self.dense1(x))
        out = out + self.dense2(out)
        return out


class BertAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-chinese", cache_dir="./model/bert")
        # 冻结原始模型参数
        for param in self.model.parameters():
            param.requires_grad = False
        # 解冻LayerNorm的参数
        for layer in self.model.encoder.layer:
            for param in layer.attention.output.LayerNorm.parameters():
                param.requires_grad = True

        # 插入adapter层
        for layer in self.model.encoder.layer:
            layer.attention.output.dense.add_module("adapter", Adapter())
            layer.output.dense.add_module("adapter", Adapter())

        self.dense = nn.Linear(in_features=768, out_features=2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0, :]
        return self.dense(cls)