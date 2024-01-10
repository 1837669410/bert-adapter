import torch

from data import load_data
from model import BertAdapter


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "")
    train_dataloader = load_data(is_train=True)
    test_dataloader = load_data(is_train=False)
    BertAdapter = BertAdapter().to(device)

    epoch = 20
    opt = torch.optim.Adam(BertAdapter.parameters(), lr=3e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    for e in range(epoch):
        BertAdapter.train()
        for step, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_dataloader):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)

            out = BertAdapter(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = loss_func(out, labels)

            loss.backward()
            opt.step()
            opt.zero_grad()

            if step % 50 == 0:
                print("epoch: [{}/{}], step: [{}/{}], loss: {:.4f}".format(e, epoch, step, len(train_dataloader), loss.item()))

        BertAdapter.eval()
        total_num = 0
        total_acc = 0
        for step, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_dataloader):
            total_num += input_ids.shape[0]
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                out = BertAdapter(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            pred = torch.softmax(out, dim=1)
            pred = torch.argmax(pred, dim=1)
            total_acc += torch.eq(pred, labels).sum().item()
        print("epoch: [{}/{}], acc: {:.4f}".format(e, epoch, total_acc / total_num))
